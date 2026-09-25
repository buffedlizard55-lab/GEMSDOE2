"""
Validate submission GeoTIFF matches required format.
Verified from: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format

Requirements:
- Same CRS as training data: EPSG:32611
- Same resolution: 100m
- Same bounds as training data, outside bounds null/nan
- Single layer, float32, values [0,1]

Template conformance (added 2026-09-25 after a real platform rejection)
------------------------------------------------------------------------
The checks above were necessary but not sufficient: a file whose finite values were all
in [0, 1] was still rejected by DrivenData with "Predicted values must be in range
[0, 1]" because 3,061 px inside the sample submission's valid (scored) region were NaN -
and NaN is not in [0, 1].  When the sample template is available this validator now
also enforces, and names:

- every px where the template is finite must be finite (else: the platform's range error)
- every px where the template is NaN must be NaN ("data outside the bounds is null or nan")
- the GDAL_NODATA tag must match the template's (masked readers rely on it)

Fix a failing file with: python scripts/sanitize_submission.py --pred FILE --write
"""

import argparse
import rasterio
import numpy as np
from pathlib import Path

def validate(pred_path, sample_path=None, training_features_path=None):
    print(f"Validating {pred_path}")
    with rasterio.open(pred_path) as src:
        print(f"  Width: {src.width}, Height: {src.height}, Count: {src.count}, Dtype: {src.dtypes}, CRS: {src.crs}, Res: {src.res}, Nodata: {src.nodata}")
        data = src.read(1)
        pred_nodata = src.nodata
        print(f"  Data min: {np.nanmin(data):.4f}, max: {np.nanmax(data):.4f}, mean: {np.nanmean(data):.4f}, nan%: {np.isnan(data).mean()*100:.2f}%")

        errors = []
        # Check CRS
        if src.crs is None:
            errors.append("CRS is None, expected EPSG:32611")
        else:
            # Check EPSG
            try:
                epsg = src.crs.to_epsg()
                if epsg != 32611:
                    errors.append(f"CRS EPSG {epsg} != 32611")
                else:
                    print("  ✓ CRS EPSG:32611")
            except:
                # Check string
                if "32611" not in str(src.crs):
                    errors.append(f"CRS {src.crs} does not contain 32611")

        # Check resolution
        res = src.res
        if abs(res[0] - 100) > 1 or abs(res[1] - 100) > 1:
            errors.append(f"Resolution {res} != 100m")
        else:
            print("  ✓ Resolution 100m")

        # Check count
        if src.count != 1:
            errors.append(f"Count {src.count} != 1")
        else:
            print("  ✓ Single band")

        # Check dtype
        if src.dtypes[0] != 'float32':
            errors.append(f"Dtype {src.dtypes[0]} != float32")
        else:
            print("  ✓ Dtype float32")

        # Check values in [0,1]
        valid = data[np.isfinite(data)]
        if len(valid) > 0:
            if valid.min() < -0.01 or valid.max() > 1.01:
                errors.append(f"Values out of [0,1]: min {valid.min()}, max {valid.max()}")
            else:
                print(f"  ✓ Values in [0,1] (min {valid.min():.4f} max {valid.max():.4f})")

        # Compare to sample if provided
        template = None
        template_nodata = None
        if sample_path and Path(sample_path).exists():
            with rasterio.open(sample_path) as sample:
                template = sample.read(1)
                template_nodata = sample.nodata
                if src.width != sample.width or src.height != sample.height:
                    errors.append(f"Size mismatch with sample: pred {src.width}x{src.height} vs sample {sample.width}x{sample.height}")
                else:
                    print(f"  ✓ Size matches sample {sample.width}x{sample.height}")
                if src.crs != sample.crs:
                    print(f"  Warning: CRS differs from sample: {src.crs} vs {sample.crs}")
                if src.transform != sample.transform:
                    print(f"  Warning: Transform differs from sample")
                else:
                    print(f"  ✓ Transform matches sample")

        # Template conformance - the check whose absence let a platform rejection through
        # ("Predicted values must be in range [0, 1]" for NaN inside the scored region).
        if template is not None and template.shape == data.shape:
            ref_valid = np.isfinite(template)
            fin = np.isfinite(data)
            nan_inside = int((ref_valid & ~fin).sum())
            finite_outside = int((~ref_valid & fin).sum())
            if nan_inside:
                errors.append(
                    f"{nan_inside} px inside the template's valid region are not finite - "
                    f"the platform rejects this with \"Predicted values must be in range "
                    f"[0, 1]\" (fix: python scripts/sanitize_submission.py --pred "
                    f"{pred_path} --write)")
            else:
                print(f"  ✓ All {int(ref_valid.sum()):,} template-valid px are finite in [0,1]")
            if finite_outside:
                errors.append(
                    f"{finite_outside} px outside the template's valid region are finite "
                    f"(spec: \"data outside the bounds is null or nan\"; fix: "
                    f"python scripts/sanitize_submission.py --pred {pred_path} --write)")
            else:
                print(f"  ✓ NaN exactly outside the template's valid region")

            def _lbl(v):
                if v is None:
                    return None
                if isinstance(v, float) and np.isnan(v):
                    return "nan"
                return v
            if _lbl(pred_nodata) != _lbl(template_nodata):
                errors.append(
                    f"Nodata tag {_lbl(pred_nodata)!r} != template's {_lbl(template_nodata)!r} "
                    f"(masked readers rely on it; fix: python scripts/sanitize_submission.py "
                    f"--pred {pred_path} --write)")
            else:
                print(f"  ✓ GDAL_NODATA matches the template ({_lbl(template_nodata)!r})")
        elif template is not None:
            errors.append(f"Template shape {template.shape} does not match pred {data.shape}")

        if training_features_path and Path(training_features_path).exists():
            with rasterio.open(training_features_path) as train:
                if src.width != train.width or src.height != train.height:
                    errors.append(f"Size mismatch with training_features: pred {src.width}x{src.height} vs train {train.width}x{train.height}")
                else:
                    print(f"  ✓ Size matches training_features")

        # spec: "data outside the bounds is null or nan" -> NaN coverage should be sane
        nan_frac = float(np.isnan(data).mean())
        print(f"  i NaN fraction {100 * nan_frac:.2f}% (NaN is expected outside the GeoDAWN footprint)")
        if nan_frac > 0.9:
            errors.append(f"{100*nan_frac:.1f}% NaN - almost nothing to score")
        finite = data[np.isfinite(data)]
        if finite.size and float(finite.max()) == 0.0:
            print("  ⚠ all finite values are 0: this is the total-fault-absence template, not a prediction")

        if errors:
            print("\n❌ Validation FAILED:")
            for e in errors:
                print(f"  - {e}")
            return False
        else:
            print("\n✅ Validation PASSED - Ready for submission!")
            print("Next: Upload to https://www.drivendata.org/competitions/306/competition-doe-gems/ via 'Submit' button")
            return True

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="predicted GeoTIFF")
    parser.add_argument("--sample", default="data/sample_submission.tif", help="sample submission")
    parser.add_argument("--train", default="data/training_features.tif", help="training features")
    args = parser.parse_args()
    # exit status matters: this gates CI and any scripted submission pipeline
    sys.exit(0 if validate(args.pred, args.sample, args.train) else 1)
