"""Read the committed real-data dev fixture back into physical units.

`data/fixture/` holds a 512x512 window cut from the OFFICIAL competition rasters
(see scripts/make_dev_fixture.py for why: the sandbox can only receive bytes through
git objects on github.com, so the full 419 MB stack cannot come here).  Features are
stored int16-quantised; this module inverts the quantisation exactly as recorded in
`fixture_manifest.json` and re-checks the manifest's own error bound, so a corrupted
or mismatched fixture fails loudly instead of silently training on garbage.

    physical = (int16 + offset_inverse) * scale + lo,   int16 == -32768 -> NaN

Measured round-trip error (recorded at creation time on the runner):
max relative error 1.56e-05 of each band's full range — i.e. the fixture is the real
data to ~5 significant figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

FIXTURE_DIR = Path("data/fixture")
MANIFEST = FIXTURE_DIR / "fixture_manifest.json"
FEATURES = FIXTURE_DIR / "fixture_features_int16.tif"
LABELS = FIXTURE_DIR / "fixture_labels.tif"


def available() -> bool:
    return MANIFEST.exists() and FEATURES.exists() and LABELS.exists()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def load_fixture(max_rel_err: float = 1e-4):
    """-> (X (H,W,C) float32 physical units w/ NaN nodata, y (H,W) float32 {0,1}, meta, manifest).

    Raises if the manifest's recorded round-trip error exceeds `max_rel_err`, or if the
    band count / grid disagree between manifest, features and labels.
    """
    if not available():
        raise FileNotFoundError(
            f"fixture not present under {FIXTURE_DIR}; it is produced on a GitHub runner by "
            "scripts/make_dev_fixture.py (see .github/workflows/fetch-competition-data.yml)"
        )
    man = load_manifest()

    worst = max(b["max_rel_roundtrip_err"] for b in man["bands"])
    if worst > max_rel_err:
        raise ValueError(f"fixture quantisation error {worst:.2e} exceeds {max_rel_err:.0e}")

    with rasterio.open(FEATURES) as src:
        q = src.read().astype(np.float64)            # (C,h,w) int16 as float
        meta = src.meta.copy()
        meta["crs"] = src.crs
        meta["transform"] = src.transform
        descs = list(src.descriptions)
        tags = [dict(src.tags(i + 1) or {}) for i in range(src.count)]

    C = q.shape[0]
    if C != man["n_bands"] or C != len(man["bands"]):
        raise ValueError(f"band count mismatch: raster {C}, manifest {man['n_bands']}")

    X = np.empty(q.shape, dtype=np.float32)
    for i, b in enumerate(man["bands"]):
        na = b["nodata_int16"]
        phys = (q[i] - b["offset"]) * b["scale"] + b["lo"]   # offset is -16000 -> subtract
        X[i] = np.where(q[i] == na, np.nan, phys).astype(np.float32)

    with rasterio.open(LABELS) as src:
        y = src.read(1)
        y = (y == 1).astype(np.float32)              # 255 is the nodata sentinel

    X = np.moveaxis(X, 0, -1)                         # (h,w,C)
    if y.shape != X.shape[:2]:
        raise ValueError(f"label grid {y.shape} != feature grid {X.shape[:2]}")
    if int(y.sum()) != man["labels"]["positive_px"]:
        raise ValueError(f"label positives {int(y.sum())} != manifest {man['labels']['positive_px']}")

    meta["band_descriptions"] = descs
    meta["band_tags"] = tags
    return X, y, meta, man


if __name__ == "__main__":
    X, y, meta, man = load_fixture()
    print(f"fixture OK: X{X.shape} y{y.shape} crs={meta['crs']} "
          f"faults={int(y.sum())} ({y.mean() * 100:.2f}%)")
    print(f"window in the official raster: {man['window']}")
    print(f"max quantisation rel. error: {max(b['max_rel_roundtrip_err'] for b in man['bands']):.2e}")
    for b in man["bands"]:
        col = np.isfinite(X[..., b["band"] - 1])
        v = X[..., b["band"] - 1][col]
        print(f"  [{b['band']:2d}] {str(b['band_name']):22s} {b['data_category']:16s} "
              f"valid={col.mean():.3f} range=[{v.min():.4g},{v.max():.4g}]")
