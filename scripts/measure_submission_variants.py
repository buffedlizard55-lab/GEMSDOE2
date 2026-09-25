#!/usr/bin/env python3
"""Measure how submission shaping changes the official metric - on HELD-OUT windows only.

Run after training:  python scripts/measure_submission_variants.py --config configs/config_recon_cpu.yaml

Writes outputs_*/submission_variants.json: a table of DTI per shaping variant, computed on
the Monte-Carlo test windows (the labels those windows carry were zeroed globally during
training, see src/dataset.make_patches), so the comparison is not fitted on training data.
This is the file that justifies `submission_shaping` in the configs - no number in this repo
is asserted without a measurement like this one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import (apply_norm_stats, fit_norm_stats, load_norm_stats,  # noqa: E402
                         load_features_and_labels, make_patches)
from src.metrics import compute_distance_weighted_tversky, score_arrays_blocked  # noqa: E402
from src.models import get_model  # noqa: E402
from src.submission_optim import dominant_thin, floor_sharpen, search_threshold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_recon_cpu.yaml")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    out_dir = Path(cfg["data"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, fmeta, lmeta, tags = load_features_and_labels(cfg["data"]["feature_path"], cfg["data"]["label_path"])
    sp = out_dir / "norm_stats.json"
    stats = load_norm_stats(sp) if sp.exists() else fit_norm_stats(X)
    Xn = apply_norm_stats(X, stats, mode=cfg["data"].get("norm_mode", "clip_zscore"))
    R = int(cfg["metric"]["R_meters"] // cfg["metric"]["resolution_m"])

    mpath = Path(a.model) if a.model else sorted(out_dir.glob("*.pt"))[0]
    manifest = {}
    if (out_dir / "manifest.json").exists():
        manifest = json.loads((out_dir / "manifest.json").read_text())
    entry = next((e for e in manifest.get("models", []) if mpath.name == e["file"]), {})
    model = get_model(arch=entry.get("arch", "unet"), encoder=entry.get("encoder", "mobilenet_v2"),
                      in_channels=int(entry.get("in_channels", Xn.shape[-1])), classes=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(mpath, map_location=device))
    model.eval()

    ps, dtis = [], []
    for mc in range(int(cfg["training"]["mc_splits"])):
        res = make_patches(Xn, y, patch_size=cfg["training"]["patch_size"], train_step=cfg["training"]["train_step"],
                           test_proportion=cfg["training"]["test_proportion"],
                           seed=mc * 10 + int(cfg["training"].get("seed", 42)), R_pixels=R)
        with torch.no_grad():
            xb = torch.from_numpy(np.ascontiguousarray(res["X_test"])).permute(0, 3, 1, 2).float()
            pr = torch.sigmoid(model(xb))[:, 0].numpy()
        tw = res["summary"]["test_windows"]
        p = int(cfg["training"]["patch_size"])
        Hp, Wp = res["summary"]["H"], res["summary"]["W"]
        pred_g = np.zeros((Hp, Wp), np.float32)
        gt_g = np.zeros((Hp, Wp), np.float32)
        for arr, (i, j) in zip(pr, tw):
            pred_g[i:i + p, j:j + p] = arr
        for arr, (i, j) in zip(res["y_test"], tw):
            gt_g[i:i + p, j:j + p] = arr
        pred_g, gt_g = pred_g[: y.shape[0], : y.shape[1]], gt_g[: y.shape[0], : y.shape[1]]
        ps.append(pred_g)
        dtis.append((f"mc{mc}", gt_g, pred_g))
        if mc > 2:
            break

    rows = []
    for name, gt_g, pred_g in dtis:
        variants = {
            "raw_probabilities (as the reference solution writes them)": pred_g,
            "reference-style threshold 0.1": np.where(pred_g > 0.1, 1.0, 0.0).astype(np.float32),
            "floor 0.3 (hard, no thinning)": floor_sharpen(pred_g, t0=0.3, hard=True),
            "floor 0.3 + dominating thin": dominant_thin(floor_sharpen(pred_g, 0.3, hard=True), R=R, p=pred_g),
            "floor 0.5 + dominating thin": dominant_thin(floor_sharpen(pred_g, 0.5, hard=True), R=R, p=pred_g),
            "soft floor 0.3 gamma=2": floor_sharpen(pred_g, t0=0.3, gamma=2.0, hard=False),
        }
        for label, q in variants.items():
            d, (tp, fp, fn) = compute_distance_weighted_tversky(np.nan_to_num(q), gt_g, R_pixels=R,
                                                                 return_components=True)
            rows.append(dict(split=name, variant=label, dti=float(d), TP_w=float(tp), FP_w=float(fp), FN_w=float(fn),
                             prob_mass=float(q.sum()), frac_positive=float((q > 0).mean())))
            print(f"{name} | {label:52s} DTI={d:.4f}  FP_w={fp:9.1f}  mass={q.sum():10.1f}  "
                  f"pos%={100 * (q > 0).mean():5.2f}")

    # threshold search on the first split (what configs' submission_shaping should be set from)
    gt0, pred0 = dtis[0][1], dtis[0][2]
    best, bestd, table = search_threshold(pred0, gt0, R=R)
    print(f"\nsearched threshold on {dtis[0][0]}: best={best} DTI={bestd:.4f}")
    out = dict(model=mpath.name, R_pixels=R, variants=rows, threshold_search=dict(best=best, dti=bestd),
               table=[dict(t0=t, thin=th, dti=d, mass=m) for t, th, d, m in table])
    (out_dir / "submission_variants.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {out_dir / 'submission_variants.json'}")


if __name__ == "__main__":
    main()
