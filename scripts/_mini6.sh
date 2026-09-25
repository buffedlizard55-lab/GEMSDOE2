#!/usr/bin/env bash
# Continue the 6-fold mini ensemble on the fixture (folds 2..5), then blend all six
# with equal vs DTI weights and record both. Calibration only - tiny windows, few epochs.
cd /home/user/GEMSDOE
OV="data.use_fixture=true training.patch_size=128 training.train_step=64 training.epochs=20 training.early_stopping_patience=6 training.max_minutes=18 model.encoder=resnet34 model.pretrained=false training.shaping_windows=4 training.shaping_grid=9 training.mc_splits=1 training.selection_mode=raw"
mkdir -p outputs_mini6
for f in 2 3 4 5; do
  echo "=== fold $f train ==="
  .venv/bin/python -m src.train --config configs/config_ci_ensemble.yaml --override $OV data.output_dir=outputs_e$f training.mc_id=$f 2>&1 | tail -3
  .venv/bin/python -m src.inference --config configs/config_ci_ensemble.yaml --override $OV data.output_dir=outputs_e$f --raw --out outputs_e$f/prob_raw.tif 2>&1 | tail -2
done
echo "=== blend: equal ==="
.venv/bin/python scripts/blend_submission.py --folds outputs_e0 outputs_e1 outputs_e2 outputs_e3 outputs_e4 outputs_e5 \
  --config configs/config_ci_ensemble.yaml --sample data/fixture/fixture_labels.tif --labels data/fixture/fixture_labels.tif \
  --out outputs_mini6/sub_equal.tif --report outputs_mini6/report_equal.json --weights equal 2>&1 | grep -E "pooled|wrote outputs|informational"
echo "=== blend: dti-weighted ==="
.venv/bin/python scripts/blend_submission.py --folds outputs_e0 outputs_e1 outputs_e2 outputs_e3 outputs_e4 outputs_e5 \
  --config configs/config_ci_ensemble.yaml --sample data/fixture/fixture_labels.tif --labels data/fixture/fixture_labels.tif \
  --out outputs_mini6/sub_dti.tif --report outputs_mini6/report_dti.json --weights dti 2>&1 | grep -E "fold weights|pooled|wrote outputs|informational"
echo MINI6_DONE
