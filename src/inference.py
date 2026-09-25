"""Inference: sliding-window ensemble + TTA -> submission GeoTIFF in the official format.

Format requirements, verbatim from the problem page (fetched 2026-09-12):
    - same projected CRS as the training data (UTM zone 11N, EPSG:32611)
    - same resolution (100 m)
    - same bounds as the training data, data outside the bounds null or nan
    - single layer, 32-bit float, values in [0, 1] = confidence of fault presence
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format

Notes on choices that matter for the score
-----------------------------------------
* Gaussian (not box) blending across overlapping windows: box averaging puts a discontinuous
  step at window borders, and the metric's FP term is a *sum over pixels*, so systematic
  edge dimming in low-GT regions is a real cost.
* TTA = 4 rotations x {identity, hflip}, each inverse-transformed before accumulation, so
  the 8 estimates are pixel-aligned.
* Predictions outside the valid data footprint are written as NaN, per the spec above.
* We output probabilities (0-1), not a thresholded mask. The metric is computed on the
  probability field: TP_w uses max_x p(x)k(d), so softening a wrong-but-nearby prediction is
  cheaper than a hard mask, and a confident 1.0 on a true pixel costs nothing on the FP term.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .dataset import (apply_norm_stats, band_names, fit_norm_stats, load_features_and_labels,
                      load_norm_stats, resolve_path, FEATURE_NAME_CANDIDATES, SAMPLE_NAME_CANDIDATES)
from .models import get_model
from .postprocess import postprocess_pipeline
from .submission_io import clean_profile, conform_to_template, write_submission


def _gaussian_weight(p: int, sigma_frac: float = 0.25) -> np.ndarray:
    y, x = np.mgrid[0:p, 0:p].astype(np.float32)
    c = (p - 1) / 2.0
    g = np.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * (sigma_frac * p) ** 2))
    return (g / g.max()).astype(np.float32)


ADOPTED_EVIDENCE_PATH = "data/evidence/emission_decision.json"


def adopted_shaping(evidence_path: str | Path = ADOPTED_EVIDENCE_PATH) -> dict | None:
    """The emission policy MEASURED on the new-fault-like population, when the record says SHIP.

    The in-domain calibration (`manifest["shaping"]`) maximises DTI against the faults the model
    TRAINED on.  Both prize phases score faults those labels do not contain, and the two populations
    disagree on sign (see scripts/decide_emission_width.py): the calibrated floor is worth 0.0247 on
    the proxy population while the measured candidate is worth 0.1365 there.  A plain
    `python -m src.inference` run therefore has to read the decision record, not the manifest, or it
    silently writes the wrong policy - which is what happened until this was wired in.

    Returns `{"t0", "thin", "dilate", "source", "evidence"}` or None when the file is absent, is
    unreadable, or does not end in a SHIP verdict (in that case the caller keeps the old behaviour
    and says so in the summary/TIFF tags).
    """
    path = Path(evidence_path)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    verdict = (d.get("verdict") or {})
    if not str(verdict.get("conclusion", "")).startswith("SHIP"):
        return None
    name = str((verdict.get("best_measured_candidate") or {}).get("policy", ""))
    m = re.search(r"t0_([0-9.]+)_width([0-9]+)px", name)
    if not m:
        return None
    return {"t0": float(m.group(1)), "thin": True, "dilate": int(m.group(2)),
            "source": "adopted_measured_policy", "evidence": str(path)}


def effective_shaping(manifest_shaping: dict | None, cfg_shaping: dict | None,
                      adopted: dict | None) -> dict:
    """Decide which shaping a submission write uses, and record where it came from.

    Precedence, in order:
      1. an explicit `inference.submission_shaping.t0` in the config (an experiment that says what
         it wants) - kept, but tagged `explicit_config_override` and reported as having overridden a
         measured policy when one exists, so it can never pass for an adopted default;
      2. the adopted measured policy, when the decision record says SHIP (the default path: the
         config ships with nulls precisely so this is what happens);
      3. the manifest's in-domain calibration (the historical behaviour, only reached when no
         measurement exists yet) - tagged `in_domain_manifest_calibration`.
    """
    shp = dict(manifest_shaping or {})
    overrides = {k: v for k, v in (cfg_shaping or {}).items() if v is not None}
    explicit_floor = "t0" in overrides
    shp.update(overrides)
    out = {"t0": shp.get("t0"), "thin": shp.get("thin", True), "hard": shp.get("hard", True),
           "gamma": shp.get("gamma", 1.0), "dilate": int(shp.get("dilate") or 0),
           "source": "in_domain_manifest_calibration", "adopted": None,
           "in_domain_control": {"t0": (manifest_shaping or {}).get("t0"),
                                 "thin": (manifest_shaping or {}).get("thin")}}
    if explicit_floor:
        out["source"] = "explicit_config_override"
        out["overrode_adopted"] = adopted
        return out
    if adopted:
        out.update({"t0": adopted["t0"], "thin": bool(adopted["thin"]),
                    "dilate": int(adopted["dilate"]), "source": adopted["source"],
                    "adopted": adopted})
    return out


def _tta_variants(x: torch.Tensor, tta: bool):
    """yields (transformed_batch, inverse_fn). x: (B,C,H,W)."""
    yield x, lambda t: t
    if not tta:
        return
    for k in (1, 2, 3):
        yield torch.rot90(x, k, dims=[2, 3]), (lambda t, k=k: torch.rot90(t, -k, dims=[2, 3]))
    xf = torch.flip(x, dims=[3])
    yield xf, lambda t: torch.flip(t, dims=[3])
    for k in (1, 2, 3):
        yield torch.rot90(xf, k, dims=[2, 3]), (lambda t, k=k: torch.flip(torch.rot90(t, -k, dims=[2, 3]), dims=[3]))


@torch.no_grad()
def sliding_window_inference(model, X, patch_size, overlap=0.5, batch_size=16, device="cpu",
                             tta=True, gaussian=True, sigma_frac=0.25):
    """X: (H,W,C) float normalised -> (H,W) probability map."""
    model.eval()
    H, W, C = X.shape
    stride = max(1, int(round(patch_size * (1 - overlap))))
    ys = list(range(0, max(1, H - patch_size + 1), stride))
    xs = list(range(0, max(1, W - patch_size + 1), stride))
    if ys and ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs and xs[-1] + patch_size < W:
        xs.append(W - patch_size)
    coords = [(y, x) for y in ys for x in xs]

    Xp = X
    if H < patch_size or W < patch_size:      # tiny rasters: pad once
        Xp = np.pad(X, ((0, max(0, patch_size - H)), (0, max(0, patch_size - W)), (0, 0)))

    w_patch = _gaussian_weight(patch_size, sigma_frac) if gaussian else np.ones((patch_size, patch_size), np.float32)
    Hp, Wp = Xp.shape[:2]
    num = np.zeros((Hp, Wp), np.float32)
    den = np.zeros((Hp, Wp), np.float32)

    for b0 in tqdm(range(0, len(coords), batch_size), desc="inference", disable=len(coords) <= 1):
        blk = coords[b0:b0 + batch_size]
        batch = np.stack([Xp[i:i + patch_size, j:j + patch_size] for (i, j) in blk])
        batch = torch.from_numpy(np.ascontiguousarray(np.moveaxis(batch, -1, 1))).float().to(device)
        acc = torch.zeros(batch.shape[0], 1, patch_size, patch_size, device=device)
        n_aug = 0
        for xt, inv in _tta_variants(batch, tta):
            out = torch.sigmoid(model(xt))              # (B,1,p,p)
            acc = acc + inv(out)
            n_aug += 1
        acc = (acc / n_aug)[:, 0].cpu().numpy()         # (B,p,p)
        for n, (i, j) in enumerate(blk):
            num[i:i + patch_size, j:j + patch_size] += acc[n] * w_patch
            den[i:i + patch_size, j:j + patch_size] += w_patch
    prob = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    return prob[:H, :W]


def R_px_infer(cfg):
    return int(cfg["metric"]["R_meters"] // cfg["metric"]["resolution_m"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model-dir", default=None, help="defaults to data.output_dir")
    ap.add_argument("--out", default="submission.tif")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="write the raw ensemble probability map: skip post-processing and "
                         "shaping (used per fold so scripts/blend_submission.py can average "
                         "fold maps before shaping once, on the pooled calibration)")
    ap.add_argument("--score-against", default=None, help="label GeoTIFF -> print local DTI")
    ap.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides (same as src.train)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    for o in args.override:
        k, v = o.split("=", 1)
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        try:
            cur[parts[-1]] = json.loads(v)
        except json.JSONDecodeError:
            cur[parts[-1]] = v
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    out_dir = Path(cfg["data"]["output_dir"])
    model_dir = Path(args.model_dir) if args.model_dir else out_dir

    X, y, fmeta, lmeta, tags = load_features_and_labels(
        cfg["data"].get("feature_path"), cfg["data"].get("label_path"),
        require_labels=bool(args.score_against),
        use_fixture=bool(cfg["data"].get("use_fixture")),
        use_external_dem=bool(cfg["data"].get("use_external_dem", False)),
        external_dem_path=cfg["data"].get("external_dem_path"),
    )
    stats_path = out_dir / "norm_stats.json"
    if stats_path.exists():
        stats = load_norm_stats(stats_path)
        print(f"normalisation stats: {stats_path} (train-time, reused)")
    else:
        stats = fit_norm_stats(X, tuple(cfg["data"].get("clip_percentile", (1.0, 99.0))))
        print("WARNING: no norm_stats.json - re-fitted on this raster (only valid if the "
              "features are identical to training-time, i.e. same GeoTIFF)")
    Xn = apply_norm_stats(X, stats, mode=cfg["data"].get("norm_mode", "clip_zscore"))
    valid = np.isfinite(X).any(axis=-1)                     # footprint of real data
    H_grid, W_grid = X.shape[:2]
    # Free the un-normalised stack (see the same change in src/train.py): at the full
    # 19-band GeoDAWN grid it is ~933 MB that inference never reads again.
    del X

    manifest = {}
    mpath = out_dir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
    entries = manifest.get("models") or []
    ckpts = sorted(model_dir.glob("*.pt"))
    if not ckpts:
        raise SystemExit(f"no .pt checkpoints in {model_dir}; run src/train.py first")
    by_name = {e["file"]: e for e in entries}

    patch_size = cfg["training"]["patch_size"]
    good = cfg["training"].get("good_channels")
    maps = []
    for ck in ckpts:
        e = by_name.get(ck.name, {})
        in_ch = int(e.get("in_channels") or (len(good) if good else Xn.shape[-1]))
        arch = e.get("arch", cfg["model"]["architectures"][0])
        enc = e.get("encoder", cfg["model"]["segformer_encoder"] if "segformer" in arch else cfg["model"]["encoder"])
        print(f"{ck.name}: arch={arch} encoder={enc} in_ch={in_ch}")
        model = get_model(arch=arch, encoder=enc, in_channels=in_ch, classes=1, pretrained=False).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        Xin = Xn[:, :, good] if good else Xn
        p = sliding_window_inference(model, Xin, patch_size=patch_size,
                                      overlap=cfg["inference"]["overlap"],
                                      batch_size=cfg["inference"]["batch_size"], device=device,
                                      tta=(not args.no_tta) and cfg["inference"].get("tta", True),
                                      gaussian=cfg["inference"].get("blend_mode", "gaussian") == "gaussian")
        maps.append((p, float(e.get("dti", 0.0))))

    if len(maps) > 1 and cfg["inference"].get("ensemble_weights") == "dti" and all(w > 0 for _, w in maps):
        ws = np.array([w for _, w in maps], dtype=np.float64)
        ws = np.exp((ws - ws.max()) * 8.0); ws /= ws.sum()
        final = sum(w * p for w, (p, _) in zip(ws, maps)).astype(np.float32)
        print(f"DTI-weighted ensemble, weights={np.round(ws, 3).tolist()}")
    else:
        final = np.mean([p for p, _ in maps], axis=0).astype(np.float32)

    # --raw mode: fold-level probability map, untouched by post-processing or shaping.
    # scripts/blend_submission.py averages fold maps first, then shapes ONCE with the
    # calibration pooled across folds (one thinning pass, not six).
    if args.raw:
        shp, _binary = {}, None
    else:
        final, _binary = postprocess_pipeline(final, cfg.get("postprocess", {}))
        # metric-aware shaping (floor + distance-R dominating thinning + emission width).
        # The policy that SHIPS is the one measured on the new-fault-like population
        # (data/evidence/emission_decision.json); the manifest's in-domain calibration is only
        # reached when no such measurement exists.  See effective_shaping() for the precedence.
        shp = effective_shaping(manifest.get("shaping"),
                                cfg["inference"].get("submission_shaping"),
                                adopted_shaping())
        if shp["source"] == "adopted_measured_policy":
            print(f"ADOPTED measured emission policy: t0={shp['t0']:g} thin={shp['thin']} "
                  f"dilate={shp['dilate']}px  (evidence: {shp['adopted']['evidence']}; the in-domain "
                  f"calibration would have used t0={shp['in_domain_control']['t0']})")
        elif shp["source"] == "explicit_config_override":
            print(f"WARNING submission_shaping came from the CONFIG (t0={shp['t0']}), not from a "
                  f"measurement - this is an experiment path, and the written tags say so.")
            if shp.get("overrode_adopted"):
                a = shp["overrode_adopted"]
                print(f"WARNING it OVERRIDES the adopted measured policy (t0={a['t0']:g}, "
                      f"width {a['dilate']}px) recorded in {a['evidence']}.")
        elif shp["t0"] is None:
            print("NOTE no adopted policy and no calibrated floor found: the submission is being "
                  "written UNSHAPED.")
    pre_mass = float(np.nansum(final))          # machine-checkable record for the docs table
    if not args.raw and (shp.get("t0") is not None or shp.get("enabled")):
        from .submission_optim import optimize_submission
        pre = pre_mass
        final = optimize_submission(final, R=R_px_infer(cfg), t0=float(shp.get("t0", 0.3)),
                                    thin=bool(shp.get("thin", True)), hard=bool(shp.get("hard", True)),
                                    gamma=float(shp.get("gamma", 1.0)),
                                    dilate=int(shp.get("dilate") or 0))
        print(f"submission shaping {shp}: probability mass {pre:.0f} -> {float(np.nansum(final)):.0f}")
    final = np.clip(final, 0.0, 1.0).astype(np.float32)
    final[~valid] = np.nan                                  # "outside the bounds is null or nan"

    # ---- write with the sample submission's grid when available (authoritative template) --
    try:
        sample = resolve_path(cfg["data"].get("sample_submission_path"), SAMPLE_NAME_CANDIDATES)
    except FileNotFoundError:
        sample = None
    if sample:
        with rasterio.open(sample) as src:
            h, w, crs, tr, dtype = src.height, src.width, src.crs, src.transform, src.dtypes[0]
        print(f"grid from sample submission {sample}: {w}x{h} {crs} {dtype}")
    else:
        h, w = y.shape if y is not None else X.shape[:2]
        crs, tr = (lmeta or fmeta)["crs"], (lmeta or fmeta)["transform"]
    if final.shape != (h, w):
        # crop/pad rather than resample: the spec demands identical bounds
        print(f"WARNING resizing {final.shape} -> {(h, w)} (crop/pad, no resampling)")
        buf = np.full((h, w), np.nan, np.float32)
        hh, ww = min(h, final.shape[0]), min(w, final.shape[1])
        buf[:hh, :ww] = final[:hh, :ww]
        final = buf

    # Always write through the fail-loud writer.  Do not copy the sample's block geometry
    # and force TILED=YES: the official template is striped and its width (3292) is not a
    # legal TIFF tile width.  That exact combination previously left a tiny unreadable
    # stub while a piped workflow reported success; clean_profile() removes the stale
    # layout and write_submission() reads the bytes back before returning.
    if sample:
        with rasterio.open(sample) as src:
            source_profile = src.profile.copy()
            template = src.read(1)
            sample_nodata = src.nodata
        # Template conformance (2026-09-25): `final[~valid]` above NaNs the FEATURE
        # footprint, which does not exactly equal the sample's valid region - NaN inside
        # that region is what the platform rejects as "Predicted values must be in range
        # [0, 1]".  Align to the template, keep its nodata tag, report what changed.
        final, conf = conform_to_template(final, template)
        if conf["filled_inside"] or conf["masked_outside"] or conf["clipped"]:
            print(f"template conformance: filled {conf['filled_inside']} NaN px inside the "
                  f"sample's valid region, masked {conf['masked_outside']} px outside it, "
                  f"clipped {conf['clipped']} px to [0,1]")
        profile = clean_profile(source_profile, height=h, width=w, crs=crs, transform=tr,
                                dtype="float32", nodata=sample_nodata)
    else:
        profile = clean_profile(None, height=h, width=w, crs=crs, transform=tr,
                                dtype="float32", nodata=float("nan"))
    written = write_submission(
        args.out, final, profile,
        band_description="fault-presence probability (distance-weighted Tversky submission)",
        tags={"source": "GEMSDOE ensemble", "models": ";".join(c.name for c in ckpts),
              "n_models": len(ckpts), "tta": not args.no_tta,
              # Provenance, mirroring scripts/blend_submission.py: a written submission says which
              # policy shaped it and where that policy came from, so "measured" is distinguishable
              # from "the calibration happened to agree" by reading the raster alone.
              "shaping_t0": str(shp.get("t0")), "shaping_thin": str(bool(shp.get("thin", True))),
              "shaping_dilate": str(int(shp.get("dilate") or 0)),
              "shaping_source": str(shp.get("source")),
              "shaping_evidence": str((shp.get("adopted") or {}).get("evidence", ""))},
    )
    print(f"wrote {args.out}  valid_px={written['finite_px']}/{written['total_px']} "
          f"nonzero={written['nonzero_px']} mass={written['prob_mass']:.1f} "
          f"tiled={written['tiled']} blocks={written['block_shapes']}")

    # Run summary next to the submission: docs/results.html quotes these numbers, and
    # scripts/audit_docs.py re-derives that row from this file so a stale claim fails CI.
    summary = {
        "out": str(args.out),
        "grid": {"width": int(w), "height": int(h), "crs": str(crs)},
        "shaping_applied": bool(shp.get("t0") is not None or shp.get("enabled")),
        "shaping": {k: shp.get(k) for k in ("t0", "thin", "hard", "gamma")},
        "shaping_source": shp.get("source"),
        "shaping_dilate": int(shp.get("dilate") or 0),
        "shaping_adopted": shp.get("adopted"),
        "shaping_in_domain_control": shp.get("in_domain_control"),
        "prob_mass_pre_shaping": round(pre_mass, 1),
        "prob_mass_post_shaping": round(float(np.nansum(final)), 1),
        "nonzero_px": int(np.count_nonzero(np.nan_to_num(final))),
        "finite_px": int(np.isfinite(final).sum()),
        "total_px": int(final.size),
        "mean": round(float(np.nanmean(final)), 6),
        "min": round(float(np.nanmin(final)), 6),
        "max": round(float(np.nanmax(final)), 6),
        "n_models": len(ckpts), "models": [c.name for c in ckpts],
        "tta": bool(not args.no_tta),
    }
    spath = Path(args.out).with_name("inference_summary.json")
    spath.write_text(json.dumps(summary, indent=1))
    print(f"wrote {spath}  (mass {summary['prob_mass_pre_shaping']:.0f} -> "
          f"{summary['prob_mass_post_shaping']:.0f})")

    if args.score_against and y is not None:
        from .metrics import score_arrays_blocked
        R = int(cfg["metric"]["R_meters"] // cfg["metric"]["resolution_m"])
        d, (tp, fp, fn) = score_arrays_blocked(np.nan_to_num(final), (y > 0.5).astype(np.float32), R_pixels=R)
        print(f"LOCAL DTI vs training labels = {d:.4f} (TP={tp:.0f} FP={fp:.0f} FN={fn:.0f}) "
              "- optimistic: these are the labels the model was trained on")


if __name__ == "__main__":
    main()
