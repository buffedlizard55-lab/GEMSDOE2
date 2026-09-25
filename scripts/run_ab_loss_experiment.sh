#!/usr/bin/env bash
# A/B experiment: which loss wins on held-out DTI?  (CPU, reconstructed public data.)
# Results -> outputs_ab/summary.txt  (every number in the docs about this experiment
# comes from that file; nothing here is asserted before it is measured).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/user/.venv/bin/python}
mkdir -p outputs_ab
: > outputs_ab/summary.txt
for L in combined_dw dw_tversky plain_tversky; do
  echo "===== loss=$L =====" | tee -a outputs_ab/summary.txt
  $PY -m src.train --config configs/config_recon_cpu.yaml \
      --override training.mc_splits=3 training.epochs=4 training.loss=$L \
                 data.output_dir=outputs_ab/$L model.pretrained=false \
                 model.architectures="[\"unet\"]" model.encoder=mobilenet_v2 \
                 inference.batch_size=8 training.batch_size=8 \
    2>&1 | grep -E "pooled shaping|mean best DTI|Traceback|Error" | tee -a outputs_ab/summary.txt
done
echo "done" >> outputs_ab/summary.txt
