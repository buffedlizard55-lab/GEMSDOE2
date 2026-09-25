#!/usr/bin/env bash
# One-off calibration run (sandbox, 2 cores): 2-fold mini ensemble on the REAL-data
# fixture to prove the ensemble->blend->shaping path before the runner job executes it.
cd /home/user/GEMSDOE
OV="data.use_fixture=true training.patch_size=128 training.train_step=64 training.epochs=20 training.early_stopping_patience=6 training.max_minutes=18 model.encoder=resnet34 model.pretrained=false training.shaping_windows=4 training.shaping_grid=9 training.mc_splits=1 training.selection_mode=raw"
set -x
mkdir -p outputs_mini
rm -rf outputs_e0 outputs_e1
.venv/bin/python -m src.train --config configs/config_ci_ensemble.yaml --override $OV data.output_dir=outputs_e0 training.mc_id=0 2>&1 | tee /tmp/ens_f0.log
.venv/bin/python -m src.inference --config configs/config_ci_ensemble.yaml --override $OV data.output_dir=outputs_e0 --raw --out outputs_e0/prob_raw.tif 2>&1 | tee -a /tmp/ens_f0.log
.venv/bin/python -m src.train --config configs/config_ci_ensemble.yaml --override $OV data.output_dir=outputs_e1 training.mc_id=1 2>&1 | tee /tmp/ens_f1.log
.venv/bin/python -m src.inference --config configs/config_ci_ensemble.yaml --override $OV data.output_dir=outputs_e1 --raw --out outputs_e1/prob_raw.tif 2>&1 | tee -a /tmp/ens_f1.log
.venv/bin/python scripts/blend_submission.py --folds outputs_e0 outputs_e1 --config configs/config_ci_ensemble.yaml --sample data/fixture/fixture_labels.tif --labels data/fixture/fixture_labels.tif --out outputs_mini/submission.tif --report /tmp/mini_ensemble_report.json 2>&1 | tee /tmp/ens_blend.log
echo MINI_ENSEMBLE_DONE
