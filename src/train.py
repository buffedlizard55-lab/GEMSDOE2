"""Training for the GEMS Prize - Monte-Carlo CV ensemble trained on the *actual* metric.

Design decisions, each traceable to a verified source:
* metric = distance-weighted Tversky (alpha=0.2, beta=0.8, R=300 m) - problem page
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
  -> we optimise a differentiable version of that exact metric (src/losses.py), plus BCE for
     dense gradients, instead of the reference solution's plain TverskyLoss.
* Monte-Carlo cross-validation (5 splits) and 128-px patches are the reference solution's
  design (cell 16: MC=5, patch_size=128, test_proportion=0.5, batch 32, epochs 5,
  AdamW lr=1e-4, resnet18 U-Net).  We keep the *structure* (so results are comparable) and
  raise capacity/schedule for the leaderboard run in configs/config.yaml.
* model selection uses the metric computed on the *stitched global* held-out test patches
  (not a mean of per-patch scores), because DTI is not decomposable over patches: the
  official scorer evaluates one continuous raster.
* every artefact needed to reproduce is written to outputs/manifest.json
  (Official Rules 3.5: assets must "sufficiently reproduce the winning results").
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (FaultDataset, band_names, load_features_and_labels, load_norm_stats,
                      make_patches, apply_norm_stats, fit_norm_stats, save_norm_stats)
from .losses import CombinedLoss, TverskyLoss, DistanceWeightedTverskyLoss
from .metrics import (GtContext, compute_distance_weighted_tversky, score_arrays_blocked,
                      score_within_mask)
from .models import count_params, get_model
from .submission_optim import search_threshold, shaping_thresholds, optimize_submission


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_patches(model, X, device, batch_size=16):
    model.eval()
    out = []
    ds = torch.utils.data.TensorDataset(torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).float())
    for (xb,) in DataLoader(ds, batch_size=batch_size):
        logits = model(xb.to(device))
        out.append(torch.sigmoid(logits)[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), np.float32)


def stitch(patches, origins, shape, crop=None):
    """Place non-overlapping patches back into a global map.

    `shape` is the *padded* grid (windows are aligned to it); `crop` trims back to the
    original raster extent, mirroring the reference solution's
    `y_final = y_combined[:y_orig.shape[0], :y_orig.shape[1]]`.
    """
    out = np.zeros(shape, np.float32)
    p = patches.shape[1]
    for arr, (i, j) in zip(patches, origins):
        out[i:i + p, j:j + p] = arr
    if crop is not None:
        out = out[: crop[0], : crop[1]]
    return out


def _compact_window_subset(tw, gt_stack, nmax, patch):
    """Pick up to `nmax` test windows whose bounding box is COMPACT and fault-bearing.

    The shaping search scores the bbox of these windows; row-major-consecutive runs
    of a non-overlapping grid are spatial neighbours, so a run that contains labels
    with the smallest bbox is a cheap, representative calibration crop (instead of a
    scatter of windows whose bbox is the whole 3292x3730 raster -> EDT cost x ~20).
    """
    n = len(tw)
    if nmax <= 0 or n <= nmax:
        return tw
    best = None
    for s0 in range(0, n - nmax + 1):
        sub = tw[s0:s0 + nmax]
        if int(gt_stack[s0:s0 + nmax].sum()) == 0:
            continue
        rows = [i for i, _ in sub]
        cols = [j for _, j in sub]
        area = (max(rows) - min(rows) + patch) * (max(cols) - min(cols) + patch)
        if best is None or area < best[0]:
            best = (area, sub)
    return best[1] if best else tw[:nmax]


def heldout_maps(model, res, y_shape, device, cfg, R):
    """Held-out (Monte-Carlo test) predictions/GT: full stitched map + search crop.

    Returns (pred_full, gt_full, pred_crop, gt_crop).  `pred_full`/`gt_full` are what the
    reported DTI uses; the cropped pair bounds the cost of the shaping grid search.
    """
    probs = predict_patches(model, res["X_test"], device, batch_size=cfg["inference"]["batch_size"])
    Hp, Wp = res["summary"]["H"], res["summary"]["W"]
    tw = res["summary"]["test_windows"]
    pred_full = stitch(probs, tw, (Hp, Wp), crop=y_shape)
    gt_full = stitch(res["y_test"], tw, (Hp, Wp), crop=y_shape)
    nmax = int(cfg["training"].get("shaping_windows", 48))
    box = _compact_window_subset(tw, res["y_test"], nmax, int(cfg["training"]["patch_size"]))
    if not box:
        return pred_full, gt_full, pred_full, gt_full
    p_ = int(cfg["training"]["patch_size"])
    y0, y1 = min(i for i, j in box), max(i for i, j in box) + p_
    x0, x1 = min(j for i, j in box), max(j for i, j in box) + p_
    return pred_full, gt_full, pred_full[y0:y1, x0:x1], gt_full[y0:y1, x0:x1]


def shaped_table(pg, gg, R, thresholds):
    """DTI of the *shaped* submission for every (t0, thin) candidate (src/submission_optim)."""
    from .submission_optim import optimize_submission
    rows = []
    for t in thresholds:
        for thin in (False, True):
            q = optimize_submission(pg, R=R, t0=float(t), thin=bool(thin))
            rows.append((float(t), bool(thin), float(compute_distance_weighted_tversky(q, gg, R_pixels=R))))
    return rows



def _union_truth(res, proxy_path: str) -> dict:
    """The combined population, restricted to the fold's held-out test-window footprint.

    Mirrors ``scripts/block_holdout_eval.py --combined-population`` exactly — combined = labels |
    (proxy code 2 & footprint), the closest local surrogate for the Phase-2 expanded truth (rules
    §3.6) — but restricted to the fold's own held-out windows so it can serve as a per-epoch model
    selection signal in ``src/train.py`` without any GPU.  The window footprint is the padded-grid
    union of ``res["summary"]["test_windows"]``; the unpadded slice of the padded maps covers the
    original raster, so a truth pixel counts iff it sits in a held-out test window.
    """
    from .dataset import load_pseudo_mask              # late import avoids a circular dep
    y = np.asarray(res["labels"], dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("union selection needs the raw labels grid; make_patches did not expose it")
    proxy_only = load_pseudo_mask(proxy_path, 2)
    if proxy_only.shape != y.shape:
        raise ValueError(f"proxy catalogue {proxy_only.shape} != label grid {y.shape}")
    labels_gt = np.nan_to_num(y, nan=0.0) > 0.5
    combined = labels_gt | proxy_only
    tw = (res.get("summary") or {}).get("test_windows") or []
    ps = int(res["summary"].get("patch_size", 128))
    H, W = combined.shape
    foot = np.zeros((H, W), dtype=bool)
    for (i, j) in tw:
        # `test_windows` are PADDED-grid origins; the scored maps are cropped back to (H, W)
        # (`heldout_maps` stitches with crop=y_shape), so clip the footprint to the grid.
        foot[i:min(i + ps, H), j:min(j + ps, W)] = True
    return dict(full=combined.astype(np.float32), mask=foot)


def _heldout_scopes(cfg, res) -> dict[str, tuple]:
    """Extra ``(truth, mask)`` pairs to score the held-out maps on (``union_selection``).

    ``truth`` is the FULL combined population (so the FP-weight EDT sees global geometry, exactly
    like ``scripts/block_holdout_eval.py``); ``mask`` is the fold's test-window footprint, so TP/FN
    are summed over union-truth pixels inside the held-out windows and FP over predictions inside
    them — the same windows the in-domain DTI is computed on.
    """
    proxy_path = str(cfg["data"].get("proxy_catalogue_path", "")).strip()
    if not bool(cfg["training"].get("union_selection")) or not proxy_path:
        return {}
    u = _union_truth(res, proxy_path)
    return {"union": (u["full"], u["mask"])}


def _score_full(pred_full, gt_full, cfg, R, scopes=None):
    dti, (tp, fp, fn) = score_arrays_blocked(np.nan_to_num(pred_full), gt_full, R_pixels=R,
                                             alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"])
    comps = dict(TP_w=tp, FP_w=fp, FN_w=fn)
    if scopes:
        pred = np.nan_to_num(pred_full)
        sc_out = {}
        for name, (truth, mask) in scopes.items():
            ctx = GtContext(truth, R_pixels=R)
            s = score_within_mask(pred, ctx, mask,
                                  alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"])
            sc_out[name] = dict(dti=s["dti"], TP_w=s["TP_w"], FP_w=s["FP_w"], FN_w=s["FN_w"],
                                n_gt=s["n_gt"], pred_px=s["pred_px"])
        comps["scopes"] = sc_out
    return dti, None, comps


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, fpw_enabled, sched=None):
    model.train()
    tot, nb = 0.0, 0
    for X, y, w in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(X)
                loss = criterion(logits, y, fp_weight=w.to(device) if fpw_enabled else None)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = criterion(logits, y, fp_weight=w.to(device) if fpw_enabled else None)
            loss.backward()
            optimizer.step()
        if sched is not None:
            sched.step()          # OneCycleLR steps per batch, not per epoch
        tot += float(loss)
        nb += 1
    return tot / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides")
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

    out_dir = Path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["training"].get("seed", 42)))
    device = pick_device()
    print(f"device={device}  config={args.config}")

    # ---------------------------------------------------------------- data ---------
    X, y, fmeta, lmeta, tags = load_features_and_labels(
        cfg["data"].get("feature_path"), cfg["data"].get("label_path"),
        use_fixture=bool(cfg["data"].get("use_fixture")),
        use_external_dem=bool(cfg["data"].get("use_external_dem", False)),
        external_dem_path=cfg["data"].get("external_dem_path"),
    )
    names = band_names(tags, X.shape[-1])
    print(f"features {X.shape} labels {y.shape}  bands={names[:4]}{'...' if len(names) > 4 else ''}")

    # PSEUDO-LABELS (external catalogue, allowed by the problem page #external-datasets).
    # Leakage safety lives in make_patches: training windows only, partition and window
    # selection on the original labels alone - see the config header for the experiment.
    pseudo_mask = None
    pl = cfg["data"].get("pseudo_label_path")
    if pl:
        from .dataset import load_pseudo_mask
        pseudo_code = int(cfg["data"].get("pseudo_code", 2))
        pseudo_mask = load_pseudo_mask(pl, pseudo_code)
        if pseudo_mask.shape != y.shape:
            raise ValueError(f"pseudo mask grid {pseudo_mask.shape} != label grid {y.shape} - "
                             f"the coded raster must be on the competition grid")
        print(f"pseudo-labels: {int(pseudo_mask.sum()):,} px (code {pseudo_code}) from {pl} "
              f"- training windows only; held-out measurement is against the labels")

    stats_path = out_dir / "norm_stats.json"
    if stats_path.exists():
        nstats = load_norm_stats(stats_path)
        print(f"reusing normalisation stats from {stats_path}")
    else:
        nstats = fit_norm_stats(X, tuple(cfg["data"].get("clip_percentile", (1.0, 99.0))))
        save_norm_stats(stats_path, nstats)
    Xn = apply_norm_stats(X, nstats, mode=cfg["data"].get("norm_mode", "clip_zscore"))
    # The un-normalised stack is not used past this point. Freeing it halves peak RSS
    # (933 MB per copy at the full 19-band GeoDAWN grid): measured 2026-09-17, the
    # 3.9 GB dev sandbox OOMs during patch extraction while both copies are alive.
    H_grid, W_grid, C_grid = X.shape
    del X

    R_px = int(cfg["metric"]["R_meters"] // cfg["metric"]["resolution_m"])
    use_fpw = bool(cfg["training"].get("use_global_fp_weight", True))

    manifest = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=cfg, band_names=names, n_bands=int(C_grid),
        feature_grid=[H_grid, W_grid], label_grid=list(y.shape),
        norm_stats_file=str(stats_path), R_pixels=R_px,
        cuda=torch.cuda.is_available(), torch=torch.__version__,
        feature_file=str(cfg["data"].get("feature_path")), models=[],
    )
    if pseudo_mask is not None:
        manifest["pseudo"] = dict(
            path=str(pl), code=int(cfg["data"].get("pseudo_code", 2)),
            weight=float(cfg["training"].get("pseudo_weight", 1.0)),
            mask_px=int(pseudo_mask.sum()),
            leakage_note=("training windows only; block partition and window selection use the "
                          "original labels alone, so the held-out measurement against the labels "
                          "cannot be a restatement of the pseudo pixels the model was shown"))

    loss_name = cfg["training"].get("loss", "combined_dw")
    criterion = (CombinedLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"], R=R_px)
                 if loss_name == "combined_dw" else
                 DistanceWeightedTverskyLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"], R=R_px)
                 if loss_name == "dw_tversky" else
                 TverskyLoss(alpha=cfg["training"]["alpha"], beta=cfg["training"]["beta"]))

    hist = []
    calib_maps = []
    # mc_id offsets the split index so a workflow can run ONE fold per job
    # (fold j: --override training.mc_id=j training.mc_splits=1).  Fold j always
    # gets the same seed/architecture regardless of how the work is parallelised.
    mc0 = int(cfg["training"].get("mc_id", 0))
    n_splits = int(cfg["training"]["mc_splits"])
    max_minutes = float(cfg["training"].get("max_minutes", 0) or 0)
    job_t0 = time.time()
    for kk in range(n_splits):
        mc = mc0 + kk
        print(f"\n=== MC split {mc + 1} (fold {kk + 1}/{n_splits}) ===")
        # ---- hold-out design --------------------------------------------------------
        # training.holdout: "random" (reference MC split, scattered windows) or
        # "spatial_blocks" (docs/DISCOVERY_PLAN.md 3a: whole 51 km blocks held out, training
        # pushed back by a collar of at least the metric's R).  The block partition is seeded
        # by training.block_seed - NOT by the fold seed - so every fold of an experiment agrees
        # on which geography belongs to which fold.
        tr = cfg["training"]
        holdout = str(tr.get("holdout", "random"))
        block_folds = int(tr.get("block_folds", 4))
        block_fold = tr.get("block_fold", None)
        block_fold = (mc % block_folds) if block_fold is None else int(block_fold)
        if not 0 <= block_fold < block_folds:
            raise SystemExit(f"training.block_fold={block_fold} outside [0,{block_folds}) - "
                             "a fold that holds out nothing would train on its own test set")
        res = make_patches(
            Xn, y, patch_size=tr["patch_size"], train_step=tr["train_step"],
            test_proportion=tr["test_proportion"], seed=mc * 10 + int(tr.get("seed", 42)),
            neg_fraction=tr.get("neg_fraction", 0.35), R_pixels=R_px,
            holdout=holdout,
            block_px=int(tr.get("block_px", 512)),
            block_folds=block_folds,
            block_fold=block_fold,
            block_buffer_px=tr.get("block_buffer_px", None),
            block_mode=str(tr.get("block_mode", "balanced")),
            block_seed=int(tr.get("block_seed", 0)),
            pseudo=pseudo_mask,
            pseudo_weight=float(tr.get("pseudo_weight", 1.0)),
        )
        print("patches:", {k: v for k, v in res["summary"].items() if not k.endswith("_windows")})
        ps = res["summary"].get("pseudo") or {}
        if ps.get("enabled"):
            print(f"pseudo: +{ps['train_px_added']:,} training-label px "
                  f"(weight {ps['weight']}) across {ps['train_windows_affected']} windows; "
                  f"partition and window selection unchanged (original labels alone)")
        ho = res["summary"].get("holdout") or {}
        if ho.get("mode") == "spatial_blocks":
            f = ho.get("fold", {})
            print(f"holdout: spatial_blocks fold {ho.get('block_fold')} of {ho.get('block_folds')} "
                  f"({ho.get('block_mode')}, block {ho.get('block_px')} px = "
                  f"{ho.get('partition', {}).get('block_km', 0)} km, buffer {ho.get('buffer_px')} px) "
                  f"-> scored {f.get('scored_px', 0):,} px, training excluded {f.get('excluded_px', 0):,} px, "
                  f"held-out fault px {f.get('fault_px', 0):,}")

        good = cfg["training"].get("good_channels")
        if good:
            res["X_train"] = res["X_train"][..., good]
            res["X_test"] = res["X_test"][..., good]
        in_ch = res["X_train"].shape[-1]

        archs = cfg["model"]["architectures"]
        arch = archs[mc % len(archs)]
        enc = cfg["model"].get("segformer_encoder", "mit_b2") if "segformer" in arch else cfg["model"]["encoder"]
        model = get_model(arch=arch, encoder=enc, in_channels=in_ch, classes=1,
                          pretrained=bool(cfg["model"].get("pretrained", False))).to(device)
        print(f"arch={arch} encoder={enc} in_ch={in_ch} params={count_params(model):,}")

        bs = cfg["training"]["batch_size"]
        train_ds = FaultDataset(res["X_train"], res["y_train"], res["fpw_train"] if use_fpw else None,
                                train=True, augment=bool(cfg["training"].get("augment", True)),
                                noise_std=float(cfg["training"].get("noise_std", 0.01)),
                                rand_crop_scale=tuple(cfg["training"].get("rand_crop_scale", (1.0, 1.0))),
                                seed=mc)
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=int(cfg["training"].get("num_workers", 0)),
                              pin_memory=(device.type == "cuda"), drop_last=False)
        # NOTE: held-out scoring goes through predict_patches() on res["X_test"] directly
        # (see heldout_maps); no test DataLoader is needed.  A TensorDataset used to be built
        # here and never iterated — a full extra copy of the test windows per fold — removed
        # 2026-09-16 (session 6).
        opt = optim.AdamW(model.parameters(), lr=float(cfg["training"]["init_lr"]),
                          weight_decay=float(cfg["training"]["weight_decay"]))
        epochs = int(cfg["training"]["epochs"])
        steps_per_epoch = max(1, len(train_dl))
        sched = None
        if cfg["training"].get("scheduler", "onecycle") == "onecycle":
            sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg["training"]["init_lr"]) * 10,
                                                   total_steps=epochs * steps_per_epoch, pct_start=0.15)
        use_amp = device.type == "cuda" and bool(cfg["training"].get("amp", True))
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        best = dict(dti=-1.0, epoch=-1)
        best_state = None
        best_maps = None
        # WHICH population early stopping / checkpointing maximise. "in_domain" (default) keeps the
        # reference solution's behaviour; "union" consumes the per-epoch union DTI (labels | proxy
        # code 2) that _heldout_scopes already logs - EXECUTIVE_SUMMARY §11 item 4. Validated on
        # the config alone before the loop so a mis-set run fails fast rather than after an epoch.
        select_on = str(cfg["training"].get("select_on", "in_domain")).strip().lower()
        if select_on not in ("in_domain", "union"):
            raise SystemExit(f"training.select_on={select_on!r} is not one of 'in_domain'/'union'")
        union_scope_configured = "union" in _heldout_scopes(cfg, res)
        if select_on == "union" and not union_scope_configured:
            raise SystemExit(
                "training.select_on='union' needs the union scope: set training.union_selection=true "
                "and data.proxy_catalogue_path to the coded proxy raster "
                "(data/evidence/proxy/proxy_catalogue.tif)")
        patience = int(cfg["training"].get("early_stopping_patience", 0) or 0)
        bad = 0
        aborted_by_budget = False
        for ep in range(epochs):
            t0 = time.time()
            if max_minutes and (t0 - job_t0) / 60.0 > max_minutes:
                print(f"time budget ({max_minutes:.0f} min) exhausted after {ep} epochs; "
                      "stopping to save the best-so-far checkpoint")
                aborted_by_budget = True
                break
            tr = train_one_epoch(model, train_dl, criterion, opt, device, scaler, use_amp,
                                 use_fpw and loss_name != "plain_tversky", sched=sched)
            pf, gf, pc, gc = heldout_maps(model, res, y.shape, device, cfg, R_px)
            scopes = _heldout_scopes(cfg, res)
            dti_g, _unused, comps = _score_full(pf, gf, cfg, R_px, scopes=scopes or None)
            # "raw" (one fixed floor+thin candidate on the compact crop - CPU-cheap, the
            # binding calibration is the pooled search below / blend_submission.py).
            # The shaped path uses shaping_thresholds() (PR #9 fix): the old
            # linspace(0.02, 0.9) grid could not reach low floors and picked its optimum
            # at the grid edge on low-contrast fields.
            sel_mode = str(cfg["training"].get("selection_mode", "shaped"))
            if sel_mode == "shaped":
                _thr = shaping_thresholds(int(cfg["training"].get("shaping_grid", 15)))
                tbl = shaped_table(pc, gc, R_px, _thr)
                dti_shaped, sh_best = (max(r[2] for r in tbl), max(tbl, key=lambda r: r[2]))
            else:
                d0 = float(compute_distance_weighted_tversky(
                    optimize_submission(pc, R=R_px, t0=0.5, thin=True), gc, R_pixels=R_px))
                dti_shaped, sh_best = d0, (0.5, True, d0)
            # ---- the selection statistic --------------------------------------------------
            # training.select_on picks WHICH population early stopping / checkpointing maximise:
            #   "in_domain" (default, unchanged): the SHAPED in-domain DTI (dti_shaped) - the object
            #                the reference solution's MC ensemble selects on;
            #   "union":     the union-population DTI logged this epoch (labels | new-fault-like proxy
            #                trace, the closest local surrogate for the Phase-2 scored population,
            #                EXECUTIVE_SUMMARY §11 item 4).  This is the OTHER half of the union
            #                signal: the per-epoch logging was the feed-in, this comparator is what
            #                consumes it.  FP_w is a sum over prediction pixels, so the union DTI is
            #                scored from the raw field on the union truth (score_within_mask), never
            #                inferred from the in-domain number.
            # A run that asks to select on the union but supplies no union truth is a configuration
            # error, not a silent fall-back to in-domain: it would claim a selection it did not make.
            union_dti = None
            if comps.get("scopes"):
                union_dti = (comps["scopes"].get("union") or {}).get("dti")
            if select_on == "union":
                if not union_scope_configured:
                    raise SystemExit(
                        "training.select_on='union' needs the union scope: set "
                        "training.union_selection=true and data.proxy_catalogue_path to the coded "
                        "proxy raster (data/evidence/proxy/proxy_catalogue.tif)")
                if union_dti is None:
                    # this fold's held-out windows contain no union truth this epoch: it cannot be
                    # the selection epoch (would select on nothing), but it is not a fatal error -
                    # a later epoch/fold may carry union truth. Warn once per occurrence.
                    print("  WARNING select_on=union but this epoch's hold-out carries no union "
                          "truth (n_gt=0); epoch not eligible to be the selected checkpoint")
                sel_stat = -1.0 if union_dti is None else float(union_dti)
            else:
                sel_stat = dti_shaped
            scope_bits = ""
            if comps.get("scopes"):
                sc = comps["scopes"].get("union") or {}
                scope_bits = f"  DTI_union={sc.get('dti','n/a')} " \
                    f"(n_gt={sc.get('n_gt','?')})" if sc.get("dti") is not None \
                    else f"  DTI_union=n/a (no union truth in hold-out)"
            print(f"epoch {ep + 1}/{epochs}  loss={tr:.4f}  DTI_raw={dti_g:.4f}  "
                  f"DTI_shaped={dti_shaped:.4f} (t0={sh_best[0]:.4g},thin={sh_best[1]})  "
                  f"TP={comps['TP_w']:.0f} FP={comps['FP_w']:.0f} "
                  f"FN={comps['FN_w']:.0f}{scope_bits}  select_on={select_on}"
                  f"(={sel_stat:.4f})  [{time.time() - t0:.0f}s]")
            hist.append(dict(mc=mc, epoch=ep + 1, loss=tr, dti_raw=dti_g, dti_shaped=dti_shaped,
                             select_on=select_on, selection_stat=float(sel_stat),
                             dti_union=(None if union_dti is None else float(union_dti)),
                             **{k: v for k, v in comps.items() if k != "scopes"}))
            if comps.get("scopes"):
                hist[-1]["scopes"] = {(n): {k: (None if v is None else float(v))
                                            for k, v in s.items()}
                                      for n, s in comps["scopes"].items()}
            # model selection on the chosen population's statistic (select_on above): that is the
            # object the checkpoint, early stopping and the manifest DTI all track.
            if sel_stat > best["dti"]:
                best = dict(dti=sel_stat, epoch=ep + 1, dti_shaped=float(dti_shaped),
                            dti_union=(None if union_dti is None else float(union_dti)))
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_maps = (pc.copy(), gc.copy())
                bad = 0
            else:
                bad += 1
                if patience and bad >= patience:
                    print(f"early stop: no {select_on}-DTI improvement for {patience} epochs")
                    break
        if cfg["training"].get("calibrate_shaping", True) and best_maps is not None:
            calib_maps.append(best_maps)            # pooled search over all splits, below
            # persist the fold's held-out crop maps so a SEPARATE blend job can pool
            # shaping calibration across folds (scripts/blend_submission.py)
            np.savez_compressed(out_dir / f"heldout_mc{mc}.npz",
                                pred=best_maps[0].astype(np.float16),
                                gt=best_maps[1].astype(np.uint8))

        if best_state is None:
            raise SystemExit(f"fold {mc}: no epoch completed (time budget?); refusing to save an empty checkpoint")
        ck = out_dir / f"model_mc{mc}_{arch}_{enc}.pt"
        torch.save(best_state, ck)
        manifest["models"].append(dict(file=ck.name, arch=arch, encoder=enc, in_channels=int(in_ch),
                                       classes=1, dti=best["dti"], epoch=best["epoch"], mc=mc,
                                       select_on=select_on,
                                       dti_shaped=best.get("dti_shaped"),
                                       dti_union=best.get("dti_union"),
                                       patch_size=cfg["training"]["patch_size"],
                                       minutes_used=round((time.time() - job_t0) / 60.0, 1),
                                       budget_hit=aborted_by_budget,
                                       holdout=ho,
                                       test_windows=res["summary"]["test_windows"]))
        print(f"saved {ck} (best DTI {best['dti']:.4f} @ epoch {best['epoch']})")

    # The normalised stack is not needed past the last split; freeing ~933 MB lowers the
    # training-loop peak on small machines (the patch/window arrays carry all the signal).
    del Xn

    # ---- pooled shaping calibration ------------------------------------------------
    # choose ONE (t0, thin) maximising the MEAN held-out DTI across all MC splits: a single
    # scalar fitted against |splits| x |windows| pixels of held-out truth, so this is far
    # from the overfitting risk of per-split tuning, and it is what inference.py applies.
    if calib_maps and cfg["training"].get("calibrate_shaping", True):
        thr = shaping_thresholds(int(cfg["training"].get("shaping_grid", 15)))
        tables = [shaped_table(pg, gg, R_px, thr) for pg, gg in calib_maps]
        mean_dti = np.mean([np.array([r[2] for r in t]) for t in tables], axis=0)
        i = int(np.argmax(mean_dti))
        raw = float(np.mean([compute_distance_weighted_tversky(pg, gg, R_pixels=R_px) for pg, gg in calib_maps]))
        t0b, thinb, _ = tables[0][i]
        manifest["shaping"] = dict(t0=float(t0b), thin=bool(thinb), hard=True, gamma=1.0,
                                   mean_heldout_dti=float(mean_dti[i]), mean_heldout_dti_raw=raw,
                                   n_splits_used=len(calib_maps),
                                   search=[dict(t0=float(r[0]), thin=bool(r[1]), mean_dti=float(mean_dti[k]))
                                           for k, r in enumerate(tables[0])],
                                   note="floor+thin by pooled held-out mean DTI; src/submission_optim.py")
        print(f"\npooled shaping: t0={t0b:.4g} thin={thinb} -> held-out DTI {mean_dti[i]:.4f} "
              f"(raw probabilities: {raw:.4f})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_dir / "train_history.json").write_text(json.dumps(hist, indent=1))
    print(f"\nTraining done. manifest -> {out_dir / 'manifest.json'}")
    if hist:
        print(f"mean best DTI across splits: "
              f"{np.mean([m['dti'] for m in manifest['models']]):.4f}  (local, test-window stitched)")


if __name__ == "__main__":
    main()
