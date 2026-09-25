#!/usr/bin/env python3
"""Generate the Recall-Union v1 submission candidate.

The previous public artifact was a backbone-plus-filtered prediction: it put the
known-fault raster in the submission and removed small components from the model
fields.  That is a reasonable catalogue reconstruction, but it is not the same
objective as Phase 1, whose hidden labels are newly identified faults.  This arm is
pre-registered for that distinction:

* the emitted field is the pixelwise UNION of two independent prediction sources --
  the deep ensemble and the CPU classical baseline;
* it deliberately does *not* copy ``labels.tif`` into the field;
* it does not discard small components, because the DTI has a 4:1 false-negative to
  false-positive penalty (alpha=0.2, beta=0.8) and a three-pixel distance tolerance;
* all local scores are diagnostics, never a leaderboard claim.  The independent
  SGMC proxy is reported as ``proxy_only`` (code 2), not the misleading union with
  the training catalogue.

This is a high-recall, new-fault-first candidate.  It is intentionally different
from ``scripts/generate_fusion_submission.py`` and from the adopted floor/thinning
artifact.  Its main risk is the larger support: the private expert labels, not any
local proxy, decide whether that risk pays off.

Official references:
- problem/competition structure:
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- metric and submission format:
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
- external data permission:
  https://www.drivendata.org/competitions/306/competition-doe-gems/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import compute_distance_weighted_tversky  # noqa: E402
from src.submission_io import clean_profile, conform_to_template, write_submission  # noqa: E402

SAMPLE = ROOT / "data/sample_submission.tif"
DEEP = ROOT / "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"
CLASSICAL = ROOT / "data/evidence/baseline/submission.tif"
LABELS = ROOT / "data/labels.tif"
PROXY = ROOT / "data/evidence/proxy/proxy_catalogue.tif"
OUT_DIR = ROOT / "data/evidence/runs/recall-union-v1"
PUBLIC_TIF = ROOT / "docs/gemsdoe2_recall_union_submission.tif"
PUBLIC_ZIP = ROOT / "docs/gemsdoe2_recall_union_submission.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def _field_from_sources() -> tuple[np.ndarray, dict]:
    """Build the field without opening labels.tif (a guard against accidental backbone leakage)."""
    if not all(p.exists() for p in (SAMPLE, DEEP, CLASSICAL)):
        missing = [str(p) for p in (SAMPLE, DEEP, CLASSICAL) if not p.exists()]
        raise FileNotFoundError("missing input(s): " + ", ".join(missing))
    template, sample_profile = _read(SAMPLE)
    deep, _ = _read(DEEP)
    classical, _ = _read(CLASSICAL)
    if deep.shape != template.shape or classical.shape != template.shape:
        raise ValueError(f"source grids {deep.shape}/{classical.shape} != template {template.shape}")
    valid = np.isfinite(template)
    # Source artifacts are binary emissions. Treat any positive finite value as a
    # candidate, but do not allow NaN/outside-footprint values into the union.
    deep_mask = np.isfinite(deep) & (deep > 0.0)
    classical_mask = np.isfinite(classical) & (classical > 0.0)
    union = valid & (deep_mask | classical_mask)
    raw = np.zeros(template.shape, dtype=np.float32)
    raw[union] = 1.0
    field, changes = conform_to_template(raw, template)
    provenance = {
        "deep_source": str(DEEP.relative_to(ROOT)),
        "classical_source": str(CLASSICAL.relative_to(ROOT)),
        "template": str(SAMPLE.relative_to(ROOT)),
        "deep_px": int((valid & deep_mask).sum()),
        "classical_px": int((valid & classical_mask).sum()),
        "overlap_px": int((valid & deep_mask & classical_mask).sum()),
        "union_px": int(union.sum()),
        "known_labels_used_to_generate_field": False,
        "component_filter": "none (recall-first arm; DTI beta=0.8 > alpha=0.2)",
        "conformance": changes,
    }
    return field, provenance


def score_diagnostics(field: np.ndarray) -> dict:
    out = {}
    if LABELS.exists():
        labels, _ = _read(LABELS)
        for name, truth in (("provided_labels", labels > 0.5),):
            dti, comp = compute_distance_weighted_tversky(field, truth, R_pixels=3,
                                                          alpha=0.2, beta=0.8,
                                                          return_components=True)
            out[name] = {"dti": float(dti), "TP_w": float(comp[0]), "FP_w": float(comp[1]),
                          "FN_w": float(comp[2]), "truth_px": int(truth.sum())}
    if PROXY.exists():
        proxy, _ = _read(PROXY)
        populations = {"sgmc_proxy_only": proxy == 2}
        if LABELS.exists():
            labels, _ = _read(LABELS)
            populations["union_labels_proxy"] = (labels > 0.5) | (proxy == 2)
        for name, truth in populations.items():
            dti, comp = compute_distance_weighted_tversky(field, truth, R_pixels=3,
                                                          alpha=0.2, beta=0.8,
                                                          return_components=True)
            out[name] = {"dti": float(dti), "TP_w": float(comp[0]), "FP_w": float(comp[1]),
                          "FN_w": float(comp[2]), "truth_px": int(truth.sum())}
    return out


def write_artifact(path: Path, field: np.ndarray, tags: dict) -> dict:
    with rasterio.open(SAMPLE) as src:
        profile = clean_profile(src.profile, height=src.height, width=src.width,
                                crs=src.crs, transform=src.transform, dtype="float32",
                                compress="lzw", tiled=False, nodata=np.nan)
    return write_submission(path, field, profile,
                            band_description="recall_union_v1", tags=tags)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PUBLIC_TIF), help="public download path")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    if not SAMPLE.exists():
        print("MISSING data/sample_submission.tif; run python scripts/assemble_data_bridge.py", file=sys.stderr)
        return 2
    try:
        field, provenance = _field_from_sources()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    report = {
        "strategy": "GEMSDOE2 Recall-Union v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/generate_recall_union_submission.py",
        "official_metric": {"R_pixels": 3, "R_meters": 300, "alpha": 0.2, "beta": 0.8},
        "provenance": provenance,
        "selection_rule": ("fixed pixelwise union of the deep ensemble and the CPU classical "
                            "baseline; no label raster and no hidden score used"),
        "limitations": [
            "SGMC proxy-only and provided-label scores are local diagnostics, not leaderboard scores.",
            "The private expert-labeled new-fault set is unavailable in this checkout.",
            "A larger support can increase recall and false-positive cost; the single final submission "
            "must be selected with that trade-off in mind.",
        ],
    }
    if args.report_only:
        report["diagnostic_scores"] = score_diagnostics(field)
        print(json.dumps(report, indent=2))
        return 0

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tags = {
        "strategy": "recall_union_v1",
        "source": "deep ensemble union CPU classical baseline",
        "known_fault_backbone": "not used",
        "component_filter": "none",
        "metric": "DTI R=3px alpha=0.2 beta=0.8",
    }
    info = write_artifact(out, field, tags)
    # Keep a byte-identical evidence copy for audit/CI; the public download is still the
    # path users click, while the evidence path is what local reports cite.
    evidence_tif = OUT_DIR / "submission.tif"
    if evidence_tif != out:
        shutil.copyfile(out, evidence_tif)
    evidence_info = dict(info)
    evidence_info["path"] = str(evidence_tif.relative_to(ROOT))
    report["artifact"] = {"path": str(out.relative_to(ROOT)), "bytes": info["bytes"],
                          "sha256": info["sha256"], "dtype": info["dtype"],
                          "finite_px": info["finite_px"], "nan_px": info["nan_px"]}
    report["evidence_artifact"] = evidence_info
    report["diagnostic_scores"] = score_diagnostics(field)
    report["elapsed_note"] = "Scores are recomputed from the final bytes by this script."
    report_path = OUT_DIR / "recall_union_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT_DIR / "submission.sha256").write_text(f"{info['sha256']}  submission.tif\n")
    with zipfile.ZipFile(OUT_DIR / "submission.zip", "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("submission.tif", out.read_bytes())
    if out == PUBLIC_TIF:
        PUBLIC_ZIP.write_bytes((OUT_DIR / "submission.zip").read_bytes())
    print(f"WROTE {out} ({info['bytes']:,} B, sha256 {info['sha256']})")
    print(f"WROTE {report_path}")
    print("NOTE: paste `Recall-Union v1 · deep+classical union · build "
          f"{info['sha256'][:8]}` into DrivenData's optional Note field")
    print(json.dumps(report["diagnostic_scores"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
