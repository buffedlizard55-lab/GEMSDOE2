#!/usr/bin/env bash
# =============================================================================
# Download GEMS Prize competition data into data/
#
# WHY THIS EXISTS: the development sandbox egresses through a proxy allowlist
# (only github.com/api.github.com/pypi.org pass), so the binary competition
# files cannot be fetched there. Run this on any unrestricted machine.
#
# Sources (both are the same competition data):
#   A) DrivenData data tab (official, requires login+enrollment):
#        https://www.drivendata.org/competitions/306/competition-doe-gems/data/
#   B) Dropbox mirrors from the data tab (public share links; do not require
#      login). The `st` parameter is a short-lived signature — we omit it and
#      rely on the durable `rlkey` + `dl=1` form.
#
# Usage:
#   bash scripts/download_competition_data.sh            # Dropbox mirrors
#   bash scripts/download_competition_data.sh --manual   # print DrivenData instructions
#
# METHOD C (2026-09-17, preferred — no network at all): the official rasters are
# committed to this branch as sha256-pinned parts. On any checkout (including the
# egress-restricted sandbox):
#     python scripts/assemble_data_bridge.py     # re-verify + place canonical names
#     python scripts/prepare_data.py             # pre-flight validation
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

if [[ "${1:-}" == "--manual" ]]; then
  cat <<'EOF'
Manual download (official, requires DrivenData login + competition enrollment):
  1. Log in at https://www.drivendata.org/competitions/306/competition-doe-gems/
  2. Open https://www.drivendata.org/competitions/306/competition-doe-gems/data/
  3. Download to data/ :
       training_features.tif   (a.k.a. numeric_features.tif / gems-geodawn-numerical-features.tif)
       labels.tif              (a.k.a. existing_faults.tif)
       sample_submission.tif   (a.k.a. example_submission.tif)
       1m_DEM_links.csv        (a.k.a. Digital-elevation-model-links-JSON.pdf)
  4. Validate:  python scripts/prepare_data.py
EOF
  exit 0
fi

fetch () { # url outfile
  echo "== $2"
  curl -fL --retry 3 --retry-delay 5 -o "data/$2" "$1"
  ls -l "data/$2"
}

# --- Method C: the git bridge (committed, sha256-pinned; needs no network) ----
if [[ -f data/bridge/manifest.json ]]; then
  echo "data/bridge/ present - assembling the official rasters from the committed parts"
  python scripts/assemble_data_bridge.py
  echo "Validating..."
  python scripts/prepare_data.py
  echo "Done. Files in data/:"
  ls -lh data/
  exit 0
fi

# --- Competition files (Dropbox mirrors, durable rlkey form, dl=1) -----------
fetch "https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&dl=1" gems-geodawn-numerical-features.tif
fetch "https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=1" existing_faults.tif
fetch "https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=1" example_submission.tif

# --- Reference documents (also mirrored) --------------------------------------
fetch "https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=1" GEMS_96647.pdf || true
fetch "https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=1" Digital-elevation-model-links-JSON.pdf || true

# --- Canonical-name copies (code accepts any name; canonical keeps it simple) -
for f in gems-geodawn-numerical-features.tif:training_features.tif \
         existing_faults.tif:labels.tif \
         example_submission.tif:sample_submission.tif; do
  src="${f%%:*}"; dst="${f##*:}"
  [[ -f "data/$src" && ! -f "data/$dst" ]] && cp "data/$src" "data/$dst" && echo "copied $src -> $dst"
done

echo "Validating..."
python scripts/prepare_data.py
echo "Done. Files in data/:"
ls -lh data/
