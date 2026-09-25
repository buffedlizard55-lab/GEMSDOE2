#!/usr/bin/env python3
"""Generate the Structure-Consensus GEMS submission candidate.

This is a deliberately different arm from the supervised CNN/backbone and the old
edge-fusion/TURBO scripts.  It does not copy the existing fault raster into the
prediction.  Instead it asks whether *independent feature families* agree on a
line-like, multi-scale discontinuity:

1. every provided band is robustly scaled on finite pixels only;
2. at three spatial scales, a band contributes gradient energy weighted by structure-
   tensor coherence (a straight, consistently oriented response scores higher than
   isotropic texture);
3. bands are aggregated within their measured data family and the six family maps
   are combined by a median/upper-quartile consensus, so six correlated magnetic
   bands cannot drown out gravity, strain, topography, seismicity, or subsurface
   evidence;
4. only a pre-registered top fraction of the valid footprint is emitted, small
   components are removed, and a one-pixel disk dilation gives the DTI's 300 m
   tolerance a bounded localization margin.

The strategy is unsupervised with respect to ``labels.tif``.  The labels are used
only for an optional diagnostic report; they never change the field that is written.
This matters because the official problem description says the scored faults are new
faults absent from the public catalogue:
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/

The output is a single-band float32 GeoTIFF conformed to sample_submission.tif.  The
candidate defaults to a binary field (0/1/NaN) so it can be shipped through the
lossless browser payload builder as well as downloaded directly.

Examples
--------
  python scripts/generate_structure_consensus_submission.py
  python scripts/generate_structure_consensus_submission.py --support-fraction 0.01 --dilation 0
  python scripts/generate_structure_consensus_submission.py --report-only

No leaderboard score is promised by this script.  ``structure_report.json`` records
all measured local diagnostics and the exact parameters instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from skimage.morphology import disk, skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import compute_distance_weighted_tversky  # noqa: E402
from src.submission_io import conform_to_template, clean_profile, write_submission  # noqa: E402

FEATURES = ROOT / "data/training_features.tif"
LABELS = ROOT / "data/labels.tif"
SAMPLE = ROOT / "data/sample_submission.tif"
OUT_DIR = ROOT / "data/evidence/runs/structure-consensus-v1"
DOCS_TIF = ROOT / "docs/gemsdoe2_structure_consensus_submission.tif"
DOCS_ZIP = ROOT / "docs/gemsdoe2_structure_consensus_submission.zip"

# Six families are determined from the official raster's band tags at runtime.  The
# names here are only a deterministic fallback for a malformed/untagged copy.
FALLBACK_FAMILIES = {
    0: "magnetic", 1: "magnetic", 2: "magnetic", 3: "strain",
    4: "gravity", 5: "magnetic", 6: "strain", 7: "strain",
    8: "magnetic", 9: "seismic", 10: "gravity", 11: "topographic",
    12: "gravity", 13: "magnetic", 14: "subsurface", 15: "seismic",
    16: "subsurface", 17: "gravity", 18: "topographic",
}
SCALES = (1.0, 3.0, 6.0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _family(tags: dict, index: int) -> str:
    """Classify a band without substring collisions such as ``magnitude`` -> magnetic."""
    name = str(tags.get("band_name", "")).lower()
    category = str(tags.get("data_category", "")).lower()
    description = str(tags.get("description", "")).lower()
    text = f"{name} {category} {description}"
    # ``mag`` must be a token/prefix in a band name or an actual magnetic category.  A
    # bare substring test incorrectly classified geod_2ndinv because its description
    # contains "magnitude".
    magnetic_name = bool(re.search(r"(?:^|[_ -])(mag|rtp|tmi|tilt)(?:[_ -]|$)", name))
    if "magnetic" in category or magnetic_name:
        return "magnetic"
    if "gravity" in category or "grav" in name or "gravity" in description:
        return "gravity"
    if "strain" in category or "geodetic" in category or any(
        token in text for token in ("shear", "dilate", "strain rate", "strain")
    ):
        return "strain"
    if "seismic" in category or "earthquake" in description or "eq_" in name:
        return "seismic"
    if "topograph" in category or "elev" in name or "elevation" in description or "slope" in name:
        return "topographic"
    if "subsurface" in category or "conduct" in name or "depth" in name:
        return "subsurface"
    return FALLBACK_FAMILIES.get(index, "other")


def _finite_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (np.abs(arr) < 1e30)


def _robust_scale(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = arr[valid]
    if vals.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(vals, (2.0, 98.0))
    if not hi > lo:
        return np.zeros(arr.shape, dtype=np.float32)
    out = np.zeros(arr.shape, dtype=np.float32)
    out[valid] = np.clip((arr[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _masked_smooth(x: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth without letting the outside-footprint fill bleed inward."""
    m = valid.astype(np.float32)
    den = ndimage.gaussian_filter(m, sigma=sigma, mode="nearest")
    num = ndimage.gaussian_filter(np.where(valid, x, 0.0).astype(np.float32), sigma=sigma,
                                  mode="nearest")
    out = num / np.maximum(den, 1e-5)
    out[~valid] = 0.0
    return out.astype(np.float32)


def _percentile_unit(x: np.ndarray, valid: np.ndarray, lo: float = 50.0,
                     hi: float = 99.5) -> np.ndarray:
    vals = x[valid & np.isfinite(x)]
    if vals.size == 0:
        return np.zeros(x.shape, dtype=np.float32)
    a, b = np.percentile(vals, (lo, hi))
    if not b > a:
        return np.zeros(x.shape, dtype=np.float32)
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(np.float32)


def _coherent_response(band: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Multi-scale gradient energy times orientation coherence in [0, 1]."""
    per_scale = []
    for sigma in SCALES:
        smooth = _masked_smooth(band, valid, sigma)
        gy, gx = np.gradient(smooth)
        # The tensor is smoothed once more to estimate the local dominant orientation.
        jxx = ndimage.gaussian_filter(gx * gx, sigma=1.0, mode="nearest")
        jyy = ndimage.gaussian_filter(gy * gy, sigma=1.0, mode="nearest")
        jxy = ndimage.gaussian_filter(gx * gy, sigma=1.0, mode="nearest")
        energy = np.sqrt(np.maximum(jxx + jyy, 0.0)).astype(np.float32)
        coherence = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy ** 2, 0.0))
        coherence = coherence / np.maximum(jxx + jyy, 1e-8)
        coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
        response = _percentile_unit(energy, valid) * (0.25 + 0.75 * coherence)
        response[~valid] = 0.0
        per_scale.append(response.astype(np.float32))
    stack = np.stack(per_scale, axis=0)
    # Max across scales finds a lineament at its natural width without summing the same
    # feature three times.  The 75th percentile is a stable tie-break for broad noise.
    response = (0.7 * np.max(stack, axis=0) + 0.3 * np.percentile(stack, 75, axis=0)).astype(np.float32)
    response[~valid] = 0.0
    return response, {"scales": list(SCALES), "response_p99": float(np.percentile(response[valid], 99))}


def compute_consensus(features_path: Path = FEATURES, sample_path: Path = SAMPLE) -> tuple[np.ndarray, dict]:
    """Return (continuous score, provenance) without reading labels."""
    with rasterio.open(sample_path) as sample:
        template = sample.read(1)
        valid = np.isfinite(template)
        shape = template.shape
    family_maps: dict[str, list[np.ndarray]] = {}
    bands = []
    with rasterio.open(features_path) as src:
        if (src.height, src.width) != shape:
            raise ValueError(f"feature grid {(src.height, src.width)} != template {shape}")
        for i in range(src.count):
            tags = dict(src.tags(i + 1) or {})
            name = tags.get("band_name") or tags.get("description") or f"band_{i + 1}"
            fam = _family(tags, i)
            raw = src.read(i + 1).astype(np.float32)
            band_valid = valid & _finite_mask(raw)
            scaled = _robust_scale(raw, band_valid)
            response, meta = _coherent_response(scaled, band_valid)
            family_maps.setdefault(fam, []).append(response)
            bands.append({"index": i + 1, "name": name, "family": fam,
                          "valid_px": int(band_valid.sum()), **meta})
            print(f"band {i + 1:02d}/{src.count}: {name} -> {fam}", flush=True)

    family_scores = {}
    for fam, maps in sorted(family_maps.items()):
        stack = np.stack(maps, axis=0)
        # Median within a family is robust to one noisy derivative band.  Scale after
        # aggregation so families enter the consensus on comparable ranks.
        raw = np.median(stack, axis=0).astype(np.float32)
        family_scores[fam] = _percentile_unit(raw, valid, 50.0, 99.5)

    fs = np.stack([family_scores[k] for k in sorted(family_scores)], axis=0)
    # A median requires half the families to agree; q75 rewards an additional independent
    # witness without allowing a single family to dominate.
    score = (0.65 * np.median(fs, axis=0) + 0.35 * np.percentile(fs, 75, axis=0)).astype(np.float32)
    score[~valid] = 0.0
    prov = {
        "algorithm": "multiscale structure-tensor gradient energy with independent-family consensus",
        "source_features": str(features_path.relative_to(ROOT)),
        "source_template": str(sample_path.relative_to(ROOT)),
        "scales_px": list(SCALES),
        "families": {k: len(v) for k, v in sorted(family_maps.items())},
        "bands": bands,
        "valid_px": int(valid.sum()),
        "score_quantiles": {str(q): float(np.percentile(score[valid], q)) for q in (50, 90, 95, 99, 99.5, 99.9)},
    }
    return score, prov


def emit(score: np.ndarray, template: np.ndarray, support_fraction: float,
         dilation: int, min_component: int) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(template)
    vals = score[valid]
    if not 0 < support_fraction < 1:
        raise ValueError("support_fraction must be between 0 and 1")
    threshold = float(np.quantile(vals, 1.0 - support_fraction))
    raw = valid & (score >= threshold)
    labels, n = ndimage.label(raw, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_component)
    filtered = raw & keep[labels]
    skeleton = skeletonize(filtered)
    if not skeleton.any():
        skeleton = filtered
    emitted = (ndimage.binary_dilation(skeleton, structure=disk(int(dilation)))
               if dilation > 0 else skeleton)
    emitted &= valid
    out = np.full(template.shape, np.nan, dtype=np.float32)
    out[valid] = emitted[valid].astype(np.float32)
    return out, {
        "support_fraction_requested": float(support_fraction),
        "threshold": threshold,
        "raw_candidate_px": int(raw.sum()),
        "components_before": int(n),
        "min_component_px": int(min_component),
        "filtered_px": int(filtered.sum()),
        "skeleton_px": int(skeleton.sum()),
        "dilation_px": int(dilation),
        "emitted_px": int(emitted.sum()),
        "emitted_fraction_of_valid": float(emitted.sum() / max(valid.sum(), 1)),
    }


def _metric_rows(fields: dict[str, np.ndarray], labels: np.ndarray | None,
                 proxy: np.ndarray | None) -> dict:
    populations = {}
    if labels is not None:
        populations["provided_labels"] = labels > 0.5
    if proxy is not None:
        # Code 2 is the part of the independent SGMC raster absent from the labels.
        populations["sgmc_proxy_only"] = proxy == 2
        if labels is not None:
            populations["union_labels_proxy"] = (labels > 0.5) | (proxy == 2)
    out = {}
    for name, field in fields.items():
        row = {}
        for pop_name, truth in populations.items():
            dti, comps = compute_distance_weighted_tversky(field, truth, R_pixels=3,
                                                           alpha=0.2, beta=0.8,
                                                           return_components=True)
            row[pop_name] = {"dti": float(dti), "TP_w": float(comps[0]),
                             "FP_w": float(comps[1]), "FN_w": float(comps[2]),
                             "truth_px": int(truth.sum()),
                             "prediction_px": int(np.count_nonzero(np.nan_to_num(field)))}
        out[name] = row
    return out


def _write(path: Path, field: np.ndarray, reference: Path, tags: dict) -> dict:
    with rasterio.open(reference) as src:
        profile = clean_profile(src.profile, height=src.height, width=src.width,
                                crs=src.crs, transform=src.transform, dtype="float32",
                                compress="lzw", tiled=False, nodata=np.nan)
    info = write_submission(path, field, profile, band_description="structure_consensus_v1",
                            tags=tags)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--support-fraction", type=float, default=0.02,
                    help="pre-registered default: top 2%% of the valid footprint")
    ap.add_argument("--dilation", type=int, default=1,
                    help="pixels grown around the thinned consensus; default 1")
    ap.add_argument("--min-component", type=int, default=5)
    ap.add_argument("--out", default=str(DOCS_TIF))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    for p in (FEATURES, SAMPLE):
        if not p.exists():
            print(f"MISSING {p}; run python scripts/assemble_data_bridge.py", file=sys.stderr)
            return 2
    t0 = time.time()
    score, provenance = compute_consensus()
    with rasterio.open(SAMPLE) as src:
        template = src.read(1)
    field, emission = emit(score, template, args.support_fraction, args.dilation, args.min_component)
    report = {
        "strategy": "GEMSDOE2 Structure-Consensus v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/generate_structure_consensus_submission.py",
        "official_problem_source": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
        "parameters": {"support_fraction": args.support_fraction, "dilation": args.dilation,
                       "min_component": args.min_component},
        "provenance": provenance,
        "emission": emission,
        "honesty": ("labels are diagnostic-only; the emitted field is computed from the 19 feature "
                    "bands and the sample mask without reading labels.tif"),
        "limitations": [
            "The hidden expert-labeled test set is unavailable locally; local DTI is a surrogate.",
            "No claim of beating the leaderboard is made until DrivenData returns a score.",
            "The one-pixel dilation is a pre-registered localization-margin choice, not tuned on the hidden set.",
        ],
    }
    labels = proxy = None
    if LABELS.exists():
        with rasterio.open(LABELS) as src:
            labels = src.read(1).astype(np.float32)
            labels[labels < 0] = 0
    proxy_path = ROOT / "data/evidence/proxy/proxy_catalogue.tif"
    if proxy_path.exists():
        with rasterio.open(proxy_path) as src:
            proxy = src.read(1)
    # A small, transparent diagnostic sweep.  It is not used to overwrite the requested field.
    sweep = {}
    for frac in (0.005, 0.01, 0.02, 0.04):
        for dil in (0, 1, 2):
            candidate, em = emit(score, template, frac, dil, args.min_component)
            sweep[f"fraction={frac:g},dilation={dil}"] = {"emission": em}
            if labels is not None:
                sweep[f"fraction={frac:g},dilation={dil}"]["provided_labels_dti"] = float(
                    compute_distance_weighted_tversky(candidate, labels > 0.5, R_pixels=3,
                                                      alpha=0.2, beta=0.8))
    report["diagnostic_sweep"] = sweep
    if labels is not None or proxy is not None:
        report["diagnostic_scores"] = _metric_rows({"selected": field}, labels, proxy)
    report["seconds"] = round(time.time() - t0, 2)
    if args.report_only:
        print(json.dumps(report, indent=2))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    tags = {"strategy": "structure_consensus_v1", "support_fraction": str(args.support_fraction),
            "dilation_px": str(args.dilation), "min_component": str(args.min_component),
            "source": "19-band feature-family consensus; no labels in field generation"}
    info = _write(out, field, SAMPLE, tags)
    # Keep a byte-identical evidence copy beside the report.  The public path is the
    # user-facing download; the evidence path is what audits and CI should cite.
    evidence_tif = OUT_DIR / "submission.tif"
    if evidence_tif != out:
        shutil.copyfile(out, evidence_tif)
    report["artifact"] = {"path": str(out.relative_to(ROOT)), "bytes": info["bytes"],
                          "sha256": info["sha256"], "dtype": info["dtype"],
                          "finite_px": info["finite_px"], "nan_px": info["nan_px"]}
    report["evidence_artifact"] = {
        "path": str(evidence_tif.relative_to(ROOT)),
        "bytes": info["bytes"],
        "sha256": info["sha256"],
    }
    report_path = OUT_DIR / "structure_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT_DIR / "submission.sha256").write_text(f"{info['sha256']}  submission.tif\n")
    # zipfile uses a fixed timestamp/date and stores the member without recompression ambiguity.
    import zipfile
    zip_path = OUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("submission.tif", out.read_bytes())
    if out == DOCS_TIF:
        DOCS_ZIP.write_bytes(zip_path.read_bytes())
    print(f"WROTE {out} ({info['bytes']:,} B, sha256 {info['sha256']})")
    print(f"WROTE {report_path}")
    print(f"NOTE: paste `Structure-Consensus v1 · support={args.support_fraction:g} · "
          f"dilation={args.dilation}px · build {info['sha256'][:8]}` into DrivenData's optional Note field")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
