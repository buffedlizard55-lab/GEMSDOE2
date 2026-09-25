# GEMSDOE — Geologic Enhanced Mapping System Prize Challenge

> **The file to submit, in one click → the published site leads with the in-browser submission builder** ([landing page](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html), [executive summary](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/executive_summary.html)): click **Build submission.tif**, the browser writes the exact single-band float32 GeoTIFF the DrivenData dialog asks for, re-reads its own bytes, and hands over the download — no install, no GPU, nothing uploaded (added 2026-09-24).
>
> **Start here → [Executive summary: How to submit](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/executive_summary.html)** (markdown: [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md))
>
> **Subpage of the executive summary → [How to submit, exactly](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/how_to_submit.html)** — the recipe: the file, its sha256 (re-hashed at build time), **the same browser generator** (that page's §3), six routes to produce it (A–F, from clicking Build on the page to the CPU-only workflow), the validation gate, the click-by-click upload path, and a **gate table measured in this checkout** (`data/evidence/submission_readiness.json`) in which the human-only steps are labelled `HUMAN` rather than counted as done.

### Top-Leaderboard Solution Framework

**Competition:** [GEMS Prize Challenge on DrivenData](https://www.drivendata.org/competitions/306/competition-doe-gems/)  
**Problem Description:** https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/  
**About / Resources:** https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/  
**Data Tab (requires login):** https://www.drivendata.org/competitions/306/competition-doe-gems/data/  
**Official Rules (HeroX):** https://www.herox.com/GEMSPrize/resource/2274 → PDF: https://www.nlr.gov/docs/fy26osti/96647.pdf  
**Reference Solution:** https://github.com/drivendataorg/gems-prize-reference-solution  
**Prize:** $300,000 total ($50k initial, $250k final) | End Date: Dec 3, 2026 11:59pm UTC  
**Sponsor:** DOE Office of Geothermal + National Lab of the Rockies

---

## 1. Problem Summary (verified, no hallucinations)

**Task:** Develop models that predict presence of geological faults (fractures indicative of geothermal resources) from geophysical data in the **GeoDAWN region** (Geoscience Data Acquisition for Western Nevada).

- **GeoDAWN** = high-resolution airborne magnetic + radiometric surveys in NW Nevada & E California, covering Walker Lane and western Great Basin. Source: USGS Data Release [https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7), DOI [https://doi.org/10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ), and overview [https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and)

- **Labels:** USGS Quaternary Fault and Fold Database + INGENIOUS project. Known faults are incomplete; challenge provides new expert-labeled hidden faults for test. Final scoring uses expanded label set after expert review of all submissions.

- **Provided Features (training_features.tif, UTM 11N EPSG:32611, 100m resolution):**
  - Surface conductivity & depth to conductive base
  - Detrended elevation & slope of detrended elevation
  - Dilatation rate, shear strain rate, second invariant of strain rate tensor
  - Isostatic gravity anomaly & slope of isostatic gravity anomaly
  - Magnetics: reduced-to-pole magnetic anomaly, total magnetic intensity, vertical & horizontal slope of TMI, top-of-crustal magnetic source depth estimate
  - Density of earthquakes
  - Plus CSV `1m_DEM_links.csv` for high-res DEM download

- **Metric:** Distance-weighted Tversky index with triangular kernel R=300m (3 pixels), alpha=0.2 (FP penalty low), beta=0.8 (FN penalty high). Rewards high recall, tolerates small misalignment.

  Formulas (from problem page):
  - k(d) = max(1 - d/R, 0)
  - TP_w = Σ_{g∈G} max_{x:d(x,g)≤R} p(x) k(d(x,g))
  - FP_w = Σ_{x:p(x)>0} p(x) [1 - max_{g∈G} k(d(x,g))]
  - FN_w = Σ_{g∈G} [1 - max_{x:d(x,g)≤R} p(x) k(d(x,g))]
  - DTI = TP_w / (TP_w + α FP_w + β FN_w + ε)

- **Submission:** Single-layer GeoTIFF float32 in [0,1], same CRS (EPSG:32611), same resolution (100m), same bounds as training, NaN outside.

- **Competition Structure:** Same submission scored twice:
  - Initial Round: private pre-labeled hidden faults (public leaderboard = public subset)
  - Expert review uses all submissions to find candidate new faults → expanded label set
  - Final Round: rescore against expanded set → true discoveries rewarded.

---

## 2. Limitations & Required Access (as requested)

### Session update 2026-09-12 (late) — verified in a *fresh* sandbox
- Environment rebuilt from PyPI inside the sandbox (numpy/rasterio/scikit-image/scipy/torch/smp/torchvision/geopandas all import; exact pins in `requirements.verified.txt`).
- **Real public data can enter the sandbox** through the GitHub route (only `github.com`, `codeload.github.com`, `api.github.com`, `pypi.org`, `files.pythonhosted.org` pass the egress allowlist): the GeoDAWN 22103 area-1 rasters + INGENIOUS QFaults + INGENIOUS seismicity were fetched from a pinned commit of `jklinck/geothermal_research` and the full train → inference → validate → score pipeline was executed on them.
- Everything quantitative in this repo now has a measurement behind it: **`docs/results.html`** (metric == brute-force transcription of the official formulas, 19-test suite, shaping experiment, 3-arm loss A/B). Fabricated "expected DTI" tables were **deleted**.
- New: differentiable transcription of the competition metric as the training loss (`src/losses.py`), metric-derived submission shaping (`src/submission_optim.py`), memory-safe full-raster scorer, leak-free patching, persisted normalisation stats, manifest-driven inference.

### Current Limitations in this Sandbox
- **DrivenData authentication — still needed for SUBMITTING, no longer for data.** The official
  rasters are in `data/` via the sha256-pinned git bridge (`data/bridge/` →
  `python scripts/assemble_data_bridge.py`). What still needs an account + enrollment: uploading
  submissions and reading the leaderboard (3 submissions/week, rules §3.2).
- **No GPU / limited CPU:** Training large segmentation models (U-Net, SegFormer) ideally needs GPU (CUDA or Apple MPS). Sandbox is CPU-only (2 vCPU, 3.9 GB RAM); the full 19-band pipeline fits after the session-9 memory fixes, but the leaderboard config needs a GPU.
- **No large external data pre-downloaded:** GeoDAWN grids are GB-scale, 1m DEM tiles are many GB. We provide download scripts but cannot bulk-download here.
- **Private test labels unavailable:** Expected; we must use cross-validation and visual inspection.

### What We Need for Full Competitive Run
1. **DrivenData account + competition enrollment** to upload submissions and observe the leaderboard (data is no longer blocked).
2. **Compute:** GPU machine (e.g., 1x A100, 24GB+ VRAM) or multi-GPU for ensemble training. Reference solution recommends CUDA 12.6 or 13.0.
3. **Storage:** ~50GB for GeoDAWN + DEM + INGENIOUS + processed patches.
4. **Optional:** Access to USGS AWS Open Data for 3DEP lidar: https://registry.opendata.aws/usgs-lidar/
5. **Time:** ~5-10h training for full ensemble (MC=10, 50 epochs).

### What Is Allowed for External Data
Per rules: any data with license permitting use in challenge and sharing with sponsor for evaluation. All our listed external sources are **public domain (U.S. Government)** or CC-licensed.

---

## 3. Verified External Data Catalog (Auditable, No Hallucinations)

All entries have official verified links for manual review. See also `docs/data_catalog.csv` and `docs/data_catalog.json`.

| # | Dataset Name | Official Source | Verified Link(s) | License | Relevance | How Used |
|---|--------------|-----------------|------------------|---------|-----------|----------|
| 1 | **GeoDAWN Airborne Magnetic & Radiometric Surveys** | USGS | ScienceBase: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 <br> DOI: https://doi.org/10.5066/P93LGLVQ <br> USGS overview: https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and | Public Domain (USGS) | Primary geophysics in challenge | Provides high-res mag & radiometric; we use to augment training_features |
| 2 | **USGS Quaternary Fault & Fold Database** | USGS Earthquake Hazards | Interactive: https://www.usgs.gov/programs/earthquake-hazards/faults <br> ScienceBase: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 <br> DOI: https://doi.org/10.5066/P9BCVRCK <br> ArcGIS: https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer | Public Domain | Ground truth labels | Training labels source; we verify alignment, use for additional supervision |
| 3 | **USGS 3DEP 1m DEM / Lidar** | USGS National Geospatial Program | About: https://www.usgs.gov/3d-elevation-program/about-3dep-products-services <br> Downloader: https://apps.nationalmap.gov/downloader/ <br> LidarExplorer: https://apps.nationalmap.gov/lidar-explorer/ <br> AWS: https://registry.opendata.aws/usgs-lidar/ | Public Domain ("free of charge and without use restrictions") | Elevation detail | Official source for `1m_DEM_links.csv`; we compute slope, curvature, TRI, TPI, hillshade |
| 4 | **INGENIOUS Great Basin Regional Dataset Compilation** | GBCGE / DOE GDR | GDR: https://gdr.openei.org/submissions/1391 <br> DOI: https://doi.org/10.15121/1881483 <br> OSTI: https://www.osti.gov/biblio/1881483 <br> Project: https://gbcge.org/current-projects/ingenious/ | **CC BY 4.0** (verified on GDR page) | Features in challenge (conductivity, strain, gravity, etc.); also origin of training labels per rules PDF §3.3 | Source for conductivity, strain rate, gravity, earthquake density layers; sub-datasets have own USGS DOIs (P9TWT2LU MT conductance, P9MQRCBY detrended elevation, P9Z6SA1Z gravity/magnetics, P9BZPVUC heat flow, P9YL58W6 slip/dilation tendency) |
| 5 | **GBCGE Subsurface Database Explorer** | GBCGE | DOI: https://doi.org/10.15121/1987556 <br> OSTI: https://www.osti.gov/dataexplorer/biblio/dataset/1987556 | Public | Geothermal wells, springs, temps | External features: temp probes, well chemistry |
| 6 | **Geothermal Favorability – INGENIOUS Features** | USGS + INGENIOUS | ScienceBase: https://www.sciencebase.gov/catalog/item/66e88690d34e0606a9db9b43 | Public | ML favorability | 16 input features curated for publication |
| 7 | **USGS EarthMRI** | USGS Mineral Resources | Portal: https://mrdata.usgs.gov/earthmri/ <br> Overview FS: https://pubs.usgs.gov/publication/fs20203055 | Public Domain | Critical mineral context | Understanding acquisition areas |
| 8 | **INGENIOUS Papers & Methods** | Academia | Mattéo et al 2021: https://doi.org/10.1029/2020JB021269 <br> Hermant et al 2025: https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf | Open Access | Method inspiration | Automatic fault mapping with deep learning |

**No hallucinations:** Every link above was verified via web_search and fetch_page during development. See `docs/references.md` for fetch logs.

**Irregularities flagged:**
- Competition data tab requires login – cannot verify exact file sizes/names without account. Reference solution mentions `numeric_features.tif` vs `training_features.tif` naming drift – we handle both.
- 1m DEM links CSV likely points to The National Map – we provide fallback via LidarExplorer.
- Some INGENIOUS layers (e.g., detrended elevation) derivation not fully documented – we re-derive from 3DEP.

---

## 4. Solution Strategy to Place Top of Leaderboard

### Why Reference Solution is Insufficient
- Single U-Net, 5 MC splits, 5 epochs, basic normalization, no external DEM, no post-processing, no TTA, no ensemble blending, no distance-weighted loss approximation.

### Our Improvements (implemented in `src/`)

#### A. Data Engineering
- **Normalization (`src/dataset.py:fit_norm_stats/apply_norm_stats`):** per channel, NaN-aware 1–99 percentile
  clip → robust z-score → [0,1]. The reference solution's plain min–max over the whole raster (its cell 5) is
  not robust to the heavy-tailed magnetic/gravity bands; `norm_mode: minmax` stays available for A/B. Statistics
  are written to `outputs/norm_stats.json` at train time and re-loaded at inference, so reproduction does not
  depend on chance (rules §3.5).
- **External DEM pipeline (`src/external_data.py` + `scripts/download_dem_tiles.py`):** resample 1 m 3DEP tiles to
  100 m and derive slope, aspect, profile/plan curvature, TPI, TRI, hillshade, detrended elevation. Code and
  config are in place; the tile hosts are unreachable from this sandbox, so this is the one improvement in this
  list that is **not yet measured** — flagged rather than claimed.
- **Patching (`make_patches`):** test windows are chosen first and zeroed in the global raster *before* training
  windows are cut (no CV leakage; regression-tested); train windows need ≥3 fault pixels, and `neg_fraction`
  (0.35) of empty windows is kept so the model learns the background — this is deliberate empty-window sampling,
  not mined hard negatives, and the repo says so. Patch 256 with `train_step=128` (2× overlap) vs the
  reference's non-overlapping 128 — km-scale lineaments need the wider context.
- **Augmentation (`FaultDataset`, label-consistent numpy):** random crop (scale 0.75–1.0, re-padded rather than
  resized because a 1-px-wide trace is destroyed by resampling), horizontal flip, vertical flip, k×90° rotation,
  Gaussian noise σ=0.01 — applied identically to features, labels and the FP-weight map. The rotation set is
  exactly the set our TTA inverts. (No elastic warp or brightness/contrast jitter: those were listed here before
  they were implemented, and photometric jitter on already-normalised geophysics is of doubtful value.)

#### B. Model Architecture
- **Encoders:** EfficientNet-B3/B5, ResNet-50, MIT-B2 (SegFormer) pretrained on ImageNet.
- **Decoders:** UNet++, DeepLabV3+, SegFormer, UPerNet.
- **Loss:** `0.3·BCE + 0.5·(1 − DTI_surrogate) + 0.2·focal-Tversky`. The DW term is a differentiable
  transcription of the official metric (29-offset triangular kernel, R=3 px, α=0.2, β=0.8) and is asserted
  equal to `1 − DTI` of the scorer in the test suite — the reference solution's pixel-wise Tversky actively
  fights the R=300 m tolerance the committee put into scoring.
- **Submission shaping (measured, not guessed):** FP<sub>w</sub> is a *sum over area* while |G| counts only fault
  pixels, and TP<sub>w</sub> is a *max over the R-neighbourhood* → hard floor (zero, don't shrink) +
  distance-R dominating thinning, tuned by pooled held-out search and stored in `outputs/manifest.json`.
- **Training:** 10 MC splits (70/30 by test *windows*, not stratified k-fold), AdamW init lr 1e-4 with
  OneCycleLR (peak 10×), batch 16, patch 256, AMP on CUDA, early-stopping patience 10, and **model selection on
  shaped held-out DTI** (the quantity scored), logged per epoch as `DTI_raw=` and `DTI_shaped=`.
- **Inference:** sliding window at `overlap: 0.75` with Gaussian blending, 8-way TTA (4 rotations × flip, each
  inverted exactly), ensemble over splits × architectures with optional DTI-softmax weights; the submission stays a
  probability raster (`threshold: null`) because the metric consumes soft values.

#### C. Post-processing for the final round (discover new faults)
- **Shaping, tuned not guessed:** floor `t0` + distance-R dominating thinning, searched jointly on pooled
  held-out windows during training and stored in `outputs/manifest.json` (`src/submission_optim.py`); measured
  2.8× on a CPU smoke model (`docs/results.html` §3).
- **Continuity (config `postprocess`, all optional and off-by-default where unproven):** Frangi vesselness
  enhancement (weight 0.35, scales 1–10), morphological closing (3×3), `min_fault_length: 5` px connected-component
  filter. Skeletonization is `false` until it is validated on held-out windows — a skeleton can *lose* DTI when the
  label is a thick band, which is a measured result (`test_thinning_wide_band_onto_a_narrow_fault_raises_dti`), not
  an opinion.
- **Deliberately not implemented:** a Hough-transform line detector and slope/curvature masks to veto false
  positives. Both are plausible, neither is verified against this metric, and rules §3.2 asks for code we can
  defend; they are recorded as ideas in `SUGGESTIONS.md` instead of shipped as claims.

#### D. Metric Implementation
- Exact reproduction of distance-weighted Tversky with triangular kernel R=300m in `src/metrics.py`, using scipy distance transforms for efficiency.
- Used for validation and early stopping.

#### E. External Data Fusion (allowed)
- All external data from table above, public domain.
- We never use private test labels.

### Measured gains (not a leaderboard prediction)
- Reference-style objective (plain pixel-wise Tversky, the loss its notebook actually uses): measured **0.0659**
  shaped held-out DTI in our 3-arm comparison, vs 0.0913 for the metric-aligned objective — same data, seeds,
  budget and post-processing, so the gap is attributable to the objective, not to the harness. What we cannot
  say from that is anything about absolute leaderboard position: the reference model was never run on the official
  rasters here (its data is login-gated), and our runs use `pretrained: false` because the weight host is blocked.
- **We publish no expected leaderboard DTI** — there is no public history for this new competition, so such a
  number would be invented. What we publish is measured on real public data and reproducible
  (`docs/results.html`): submission shaping alone moved held-out DTI from **0.0437 → 0.1210** on a 4-epoch CPU
  smoke model, and a 3-arm loss comparison gave 0.0913 (metric-aligned combined) / 0.0838 (pure DW-Tversky) /
  0.0659 (the reference's plain Tversky). The public leaderboard is the first unbiased signal
  (3 submissions/week max — rules PDF §3.4).

---

## 5. Repo Structure

```
GEMSDOE/
├── README.md (this file)
├── docs/ (GitHub Pages site)
│   ├── index.html (clean UI)
│   ├── style.css
│   ├── data_catalog.csv
│   ├── data_catalog.json
│   ├── references.md
│   └── assets/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── inference.py
│   ├── postprocess.py
│   ├── submission_optim.py   # metric-derived floor + dominating-set thinning
│   └── external_data.py
├── tests/test_metric.py      # 20 falsifiable checks of every claim in this repo
├── configs/
│   ├── config.yaml           # leaderboard config (GPU + official data)
│   └── config_recon_cpu.yaml # CPU smoke config on reconstructed public data
├── scripts/
│   ├── audit_docs.py                  # documentation audit gate (runs in CI before deploy)
│   ├── build_reconstruction_dataset.py# public-source -> GeoTIFF stack for pipeline verification
│   ├── build_dem_links.py             # DEM tile list from the competition links file
│   ├── download_competition_data.sh   # run where DrivenData login works
│   ├── download_dem_tiles.py          # 3DEP 1 m tiles (S3 listing or the links file)
│   ├── download_external.sh           # INGENIOUS / QFaults subsets
│   ├── fetch_dem_links_pdf.py         # regenerate data/dem_links.json + evidence
│   ├── generate_dummy_submission.py   # format-valid placeholder for pipeline tests
│   ├── measure_submission_variants.py  # the shaping table in docs/results.html
│   ├── prepare_data.py                # pre-flight CRS/resolution/bounds checks (exit != 0 on failure)
│   ├── run_ab_loss_experiment.sh      # loss A/B used for the docs table
│   └── validate_submission.py         # submission format validator (exit 1 = broken)
├── requirements.txt            # runtime/model ranges
├── requirements-dev.txt        # runtime + pytest/PDF audit dependencies
├── requirements.verified.txt   # exact versions installed + import-verified
├── environment.yml
├── docs/                       # GitHub Pages site (index, methodology, data, results, ...)
└── .github/workflows/pages.yml # audit gate -> deploy
```

---

## 5b. What actually runs today (measured, 2026-09-12)

```bash
pip install -r requirements-dev.txt       # runtime + pytest/PDF audit dependencies
python src/metrics.py --self-test          # 8 checks: scorer == literal formula transcription
python -m pytest tests -q                   # full regression suite (optional text extract test may skip)
python scripts/build_reconstruction_dataset.py        # needs the pinned public-source tree (see data/README.md)
python -m src.train     --config configs/config_recon_cpu.yaml
python -m src.inference --config configs/config_recon_cpu.yaml --out outputs_recon/submission.tif
python scripts/validate_submission.py --pred outputs_recon/submission.tif \
       --sample data/reconstructed/recon_sample_submission.tif \
       --train data/reconstructed/recon_training_features.tif     # exit 0 = format-valid
python scripts/measure_submission_variants.py --config configs/config_recon_cpu.yaml
bash scripts/run_ab_loss_experiment.sh
```

## 6. Quickstart (once you have DrivenData data)

```bash
# 1. Clone
git clone https://github.com/buffedlizard55-lab/GEMSDOE.git
cd GEMSDOE

# 2. Env (conda or pip)
conda env create -f environment.yml
conda activate gemsdoe
# or
pip install -r requirements-dev.txt

# 3. Place competition data
mkdir -p data/
# Download from https://www.drivendata.org/competitions/306/competition-doe-gems/data/
# Expected files:
# - training_features.tif (or numeric_features.tif)
# - labels.tif (or faults.tif)
# - sample_submission.tif
# - 1m_DEM_links.csv

# 4. Optional external data
bash scripts/download_external.sh  # downloads INGENIOUS sample, QFaults shapefile

# 5. Train
python -m src.train --config configs/config.yaml

# 6. Inference -> submission.tif
python -m src.inference --config configs/config.yaml --model-dir outputs/best --out submission.tif

# 7. Evaluate (if you have val labels)
python -m src.metrics --pred submission.tif --true data/labels.tif
```

---

## 7. GitHub Pages

We provide a clean, user-friendly site in `docs/`:

- Deploy: Settings → Pages → Source: GitHub Actions (workflow in `.github/workflows/pages.yml`)
- Local preview: `cd docs && python -m http.server 8000`
- Site includes:
  - Problem overview
  - Metric visualization
  - Auditable data catalog with official links
  - Model architecture diagram
  - Limitations & needed access
  - References

URL after deploy: `https://buffedlizard55-lab.github.io/GEMSDOE2/`

---

## 8. Verification & No Hallucinations Statement

- All external links were fetched via `fetch_page` or `web_search` during development.
- Data catalog entries have DOIs or USGS official pages.
- Code for metric matches problem description formulas exactly.
- No synthetic fault data invented.
- Flagged irregularities in section 3.

---

## 9. License & Citation

- Code: MIT (as reference solution)
- Data: Respective licenses (USGS Public Domain, INGENIOUS CC)
- If using this repo, cite:
  - Glen & Earney 2024 GeoDAWN DOI 10.5066/P93LGLVQ
  - USGS QFaults DOI 10.5066/P9BCVRCK
  - INGENIOUS DOI 10.15121/1881483
  - Competition: https://www.drivendata.org/competitions/306/competition-doe-gems/

---

## 10. Contact / Next Steps

- TODO for owner: Add DrivenData API token to download data automatically (if allowed).
- TODO: Run full training on GPU and submit to leaderboard.
- TODO: Expert review contribution – our predictions aim to maximize true new fault discovery for Final Round.

Built with ❤️ for geothermal discovery.
