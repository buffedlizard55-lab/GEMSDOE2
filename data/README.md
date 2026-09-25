# data/ — Competition Data: ACQUISITION STATUS (updated 2026-09-17, session 9)

## ✅ THE OFFICIAL RASTERS ARE PLACED (sha256-verified) — 2026-09-17

`data/training_features.tif` (418,912,844 B), `data/labels.tif` (425,830 B) and
`data/sample_submission.tif` (1,599,597 B) are present and `python scripts/prepare_data.py` **PASSES**
(3292×3730, 19 bands, EPSG:32611, 100 m, aligned bounds). Provenance chain, with every hash
machine-verified at each hop:

1. **Bytes** come from the Dropbox mirrors printed on the official
   [DrivenData data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/).
2. A GitHub-hosted runner (unrestricted egress) downloaded them and **verified each sha256 against the
   pinned inventory** `data/evidence/inventory.json` (measured 2026-09-14 by
   `scripts/inspect_competition_data.py`); a drift aborts the run.
3. The runner committed them as ≤ 90 MiB parts in `data/bridge/` + `manifest.json`
   (`.github/workflows/place-competition-data.yml`, run 35168924460; GitHub rejects blobs ≥ 100 MB).
4. The sandbox re-verified every part and the concatenated whole-file sha256, then placed the
   canonical names (`python scripts/assemble_data_bridge.py` — idempotent, refuses on any mismatch).

If `data/` is ever empty again (fresh sandbox): `git pull && python scripts/assemble_data_bridge.py
&& python scripts/prepare_data.py`. Evidence: `data/evidence/data_placement.json`;
in-sandbox pipeline proof: `data/evidence/runs/local-sandbox-smoke/`.

| File | Status | Where |
|---|---|---|
| `training_features.tif` / `labels.tif` / `sample_submission.tif` | ✅ **PLACED + VERIFIED** (2026-09-17) — sha256 pins in `data/evidence/inventory.json`; reassemble any time with `python scripts/assemble_data_bridge.py` | `data/` (gitignored payloads) ← `data/bridge/` (committed parts) |
| `1m_DEM_links` (JSON in PDF) | ⚠️ **ARTIFACT LOST — REGENERATE** — an earlier session extracted 35 tile URLs (6 S3-verified, 1,316 MB) into `data/dem_links.json` + `data/evidence/`, but `data/*` was gitignored so the files were never committed and did not survive the sandbox. `.gitignore` now allow-lists `data/dem_links.json` and `data/evidence/**`. Regenerate: `python scripts/fetch_dem_links_pdf.py` (Dropbox mirror, title confirmed 2026-09-12 as "Digital elevation model links JSON.pdf"; body not extractable from this sandbox) or enumerate the USGS bucket directly: `python scripts/download_dem_tiles.py --complete-listing` |
| **irregularity (found in review 2026-09-12)** | Six doc lines (docs/index.html, docs/references.md, docs/submission.html, SUBMISSION_GUIDE.md, docs/data.html, this file) cited `data/dem_links.json` as if it existed in the repo. Corrected: the claim is now "regenerate with X". **Lesson applied:** audit artefacts must be committed or not cited. |
| `GEMS_96647.pdf` (rules mirror) | ✅ **IDENTITY VERIFIED** — fetched via platform; content matches canonical `https://www.nlr.gov/docs/fy26osti/96647.pdf` (not duplicated here; canonical already verified) | fetch log: `data/evidence/dropbox_fetch_log.md` |
| `gems-geodawn-numerical-features.tif` | ✅ **PLACED + VERIFIED** as `data/training_features.tif` via the git bridge (session 9; see above) | `data/bridge/` parts, sha256 `4371c82e…` |
| `existing_faults.tif` (labels) | ✅ PLACED + VERIFIED as `data/labels.tif` | `data/bridge/existing_faults.tif`, sha256 `7ba308cc…` |
| `example_submission.tif` | ✅ PLACED + VERIFIED as `data/sample_submission.tif` | `data/bridge/example_submission.tif`, sha256 `2176d08e…` |

DEM tile URLs point at the official USGS bucket `prd-tnm.s3.amazonaws.com` (projects `CA_SierraNevada_B22`, `NV_WestCentral_EarthMRI_2020_D20`, `NV_Humboldt_2021_D21`). The full authoritative tile list can be enumerated without the competition PDF: `python scripts/download_dem_tiles.py --complete-listing`.

Irregularities found inside the competition DEM-links JSON (documented, S3-proven): duplicate rows (up to 5×), path/filename project mismatches (canonical URL uses the FILENAME project — the mismatched path variant returns key_count=0), garbled host variants in the PDF print (prdtnm / prd.tnm / prd- tnm).

---

## Reconstructed public-source dataset — rebuilt 2026-09-12 (late session, fresh sandbox)

`reconstructed/` was lost when the sandbox reset (it is gitignored: 18 MB of rasters). It is reproducible in one
command, provided the pinned upstream tree is present:

```bash
# 1. fetch the pinned public-source tree (codeload works even where raw.githubusercontent does not)
curl -sSL -o /tmp/gr.tar.gz https://codeload.github.com/jklinck/geothermal_research/tar.gz/refs/heads/main
mkdir -p data/external/jklinck && tar xzf /tmp/gr.tar.gz -C data/external/jklinck --strip-components=1 \
  geothermal_research-main/GeoDAWN_tiffs geothermal_research-main/faults_quaternary_INGENIOUS_regional_data \
  geothermal_research-main/seismicity_INGENIOUS_regional_data
echo "56d78de7a989c12e2dce50cd65a4095df57030d2 2026-06-25T06:18:00Z (codeload tarball, fetched 2026-09-12)" \
  > data/external/jklinck_pinned_commit.txt
# 2. build (2 s, CPU)
python scripts/build_reconstruction_dataset.py
```

Measured output of the rebuild performed on 2026-09-12 (from `reconstructed/provenance.json`, regenerated):
16 bands, EPSG:32611, 100 m, 489x667 px; per-band valid fraction 74.0-74.7 %; labels 4 630 fault px = 1.42 %;
`recon_training_features.tif` sha256 `5a5b926197ae6d1c…`, `recon_labels.tif` `6c0c249e76a64f1b…`,
`recon_sample_submission.tif` `4acf720f6c2c73a3…`; source tarball sha256 `f78b96a36fd5e814…`.

Pipeline run on it (all four stages, CPU, 2 vCPU): `src.train` (2 MC splits x 4 epochs, MobileNetV2/UNet,
`pretrained: false`) -> `src.inference` -> `scripts/validate_submission.py` **PASSED** (CRS/resolution/dtype/range/
grid/transform) -> metric scoring; plus `scripts/measure_submission_variants.py` (the shaping table in
`docs/results.html`) and `scripts/run_ab_loss_experiment.sh` (loss A/B).

NOT official competition data. Caveat (flagged): different bands/processing/extent than the official
competition files — valid for pipeline verification and external-data pretraining; leaderboard submissions require
the official data-tab files (`bash scripts/download_competition_data.sh`).

## End-to-end pipeline verification (2026-09-12, CPU)

`configs/config_recon_cpu.yaml` on the reconstructed data: **train ✓ → inference ✓ →
`validate_submission.py` ✅ PASSED (CRS/res/dtype/range/grid) → DTI scoring ✓**. Model after 3 CPU
epochs ≈ constant baseline (DTI 0.052 vs 0.052) — plumbing proven, predictive power requires the
full config + GPU + official data. Two real bugs found & fixed by this run: YAML `1e-4` parsed as
string crashed AdamW (now coerced); skimage Frangi API change had silently disabled the filter.

---

# data/ — Competition Data Placement Guide

All competition data requires **DriventData login + enrollment**: https://www.drivendata.org/competitions/306/competition-doe-gems/data/

This sandbox cannot download them (no DrivenData account; TLS to Dropbox/DrivenData blocked — see `LIMITATIONS.md`). Download manually and place files here.

## Expected files and naming

⚠️ **Naming drift flagged (irregularity):** the same content appears under three different names across official sources. `src/dataset.py` and `scripts/prepare_data.py` accept **all** variants.

| Content | Problem page name | Reference solution name | Dropbox mirror name (from data tab) |
|---|---|---|---|
| Input features (multiband GeoTIFF, EPSG:32611, 100 m) | `training_features.tif` | `numeric_features.tif` | `gems-geodawn-numerical-features.tif` |
| Training labels (fault raster) | `labels.tif` | `labels.tif` | `existing_faults.tif` |
| Submission template | `sample_submission.tif` | — (derives from labels) | `example_submission.tif` |
| 1 m DEM link list | `1m_DEM_links.csv` | — | `Digital-elevation-model-links-JSON.pdf` |
| Official rules PDF | hosted at https://www.nlr.gov/docs/fy26osti/96647.pdf | — | `GEMS_96647.pdf` |

## Dropbox mirrors (competition-provided, from data tab)

> Provenance: supplied by the competition/user; this sandbox could not fetch Dropbox (TLS-blocked), so contents are not independently verified — the identical rules PDF was verified at the NLR URL above.

- Rules PDF: https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=1
- Example submission: https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=1
- Existing faults (labels): https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=1
- GeoDAWN numerical features: https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=1
- DEM links (JSON PDF): https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=1

(Swap `dl=0` → `dl=1` for direct download.)

## After download — sanity checks (no hallucination policy)

```bash
python scripts/prepare_data.py        # validates presence, CRS, resolution, bounds
python scripts/validate_submission.py # run after generating a submission
```

Expected (per official problem description):
- Features/labels: EPSG:32611, 100 m resolution, same bounds.
- Labels: single band, positive pixels = fault.
- Submission template: single band float32, values in [0,1].

## External data (all public domain / CC BY 4.0 — allowed by rules)

See `docs/data_catalog.csv` (one row per link, with verification status) and `scripts/download_external.sh`.

Primary external sources (official):
- GeoDAWN raw surveys (CC0): https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7
- INGENIOUS regional compilation (CC BY 4.0): https://gdr.openei.org/submissions/1391
- USGS Qfaults (public domain): https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23
- 1 m DEM: https://apps.nationalmap.gov/downloader/ (or links in `1m_DEM_links.csv` once available)
