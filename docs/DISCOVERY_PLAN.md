# The scored universe: what we are actually asked to find, and how to measure it

**Status:** strategy document, 2026-09-16. Every factual claim below is tied to a source that can be
opened by hand; nothing is asserted from memory. Machine-checked quotations live in
`data/evidence/rules_quotes.json` (written by `scripts/verify_rules_quotes.py`, which extracts the
canonical rules PDF and asserts each sentence is present verbatim).

---

## 1. The fact everything else depends on

The official rules (`https://docs.nlr.gov/docs/fy26osti/96647.pdf` — the canonical copy the
competition's rules page, `https://www.drivendata.org/competitions/306/competition-doe-gems/rules/`,
points at) say three times, in three sections, that the metric is computed against **new** faults:

| section | sentence (verbatim) |
|---|---|
| §1.1 | "In Phase 1, submissions will be evaluated against a privately withheld subset of the original new fault dataset compiled by expert reviewers." |
| §1.1 | "Submissions will be reevaluated against the full, revised new fault dataset using the same distance-weighted Tversky index." |
| §3.5 | "Submissions will be automatically evaluated using a distance-weighted Tversky index against newly created fault labels…" |

and §3.3 says what the **training** labels are:

> "The training labels contain existing fault data at 100-m resolution where positively labeled
> pixels indicate fault presence. These labels were obtained from the INGENIOUS project's Great Basin
> Regional Dataset Compilation." (footnote 4 = Ayling et al. 2022, `doi:10.15121/1881483`)

So the labels we can train on and the labels we are scored on are **disjoint populations**. The
competition's own problem page states the same from the other direction
(`https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure`):
the new faults are "faults that are not contained within the current public USGS database".

**Consequence, stated plainly:** a perfect reproduction of `labels.tif` scores ≈1.0 against the
catalogue we can see and ≈0 against the target. This repository measured that row — "exact known
faults, p=1 → DTI 0.99993" (`docs/METRIC_STRATEGY.md`, `scripts/metric_strategy.py`) — which is a
calibration of the scorer, **not** a strategy. Every selection decision that uses catalogue DTI is
therefore selecting for the wrong objective.

What catalogue DTI *is* still good for:

* proving the metric implementation, the differentiable loss, the tiling and the writer agree
  (plumbing);
* proving a model has learned something about fault *morphology* rather than noise — a model at 0.19
  on unseen windows is doing something real, a model at 0.05 is at the blanket-coverage floor;
* regression detection when the pipeline changes.

What it must not be used for: choosing between models, calibrating the submission floor, or weighting
ensemble members — all three are currently calibrated on it.

## 2. What the published work on this exact task does

The competition's own About page (`https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/#additional-information`)
recommends two papers. The second is directly on task and free to read:

**Hermant, B., Kiersnowski, L., & Bellanger, M. (2025). "Using Deep Learning to Map Quaternary Faults
in Western USA."** 50th Workshop on Geothermal Reservoir Engineering, Stanford.
`https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf` — fetched and read
2026-09-16. Findings that are directly actionable here (quoted):

* **Inputs are topography, not potential fields**: "The multiband image (elevation, slope and
  satellite) of the study area is divided into tiles of the desired size... Given the resolution of
  the images used (10 m) and the size of the signature of the fault objects (10²–10⁵ m long and
  10–10² m wide), an image size of 128 x 128 pixels, i.e. 1280 x 1280 m, was chosen." Study area:
  north-central Nevada, with LIDAR coverage — the same region and the same lidar the GEMS data tab
  links to.
* **They do not train on empty tiles**, and they say why: only tiles containing at least one fault
  pixel are retained, which "makes it possible to (1) limit the imbalance in the training data ... and
  (2) limit the possibility of learning images without mapped faults when there should be some due to
  **operator observation bias**." That is the label-noise argument for our scored universe in the
  authors' own words: absence from the catalogue is not evidence of absence of a fault.
* **The catalogue itself is known to be locally wrong**: "distance between USGS Quaternary faults and
  TLS fault label can be up to 400 m" (north-central Nevada), and mapping density varies for
  administrative rather than geological reasons. This matters for our FP penalty, which is computed
  against the *new* labels: a prediction 300 m off a new fault keeps partial credit (k = 1 − d/R), so
  the metric is already tolerant of exactly this kind of offset — but only if the fault is the *new*
  one, which we cannot see.
* **Architectures**: a simplified U-Net ("siUNET", 487,297 parameters) and a deeper "FaultSEG" model
  (~100× more parameters) with per-band standardisation `X' = (X − μ)/σ`. Both are conventional
  segmentation networks — i.e. no exotic architecture is needed; data and labels are the bottleneck.

**Mattéo, L., Manighetti, I., Tarabalka, Y., Gaucel, J.-M., van den Ende, M., Mercier, A., et al.
(2021). "Automatic fault mapping in remote optical images and topographic data with deep learning."**
JGR Solid Earth 126, e2020JB021269, `https://doi.org/10.1029/2020JB021269` (also recommended by the
About page; paywalled abstract only in this sandbox, cited here as the About page cites it).

The About page additionally lists the classical toolbox the sponsors had in mind — "edge detection,
Hough transforms, and deep learning on seismic or topographic datasets" — as legitimate approaches.

## 3. Measurement protocol (a and b implemented; b2 built and dispatched)

We cannot see the scored faults, so the honest options are:

**(a) Spatial block hold-out as the selection signal — IMPLEMENTED 2026-09-18.** Split the GeoDAWN
grid into large blocks, train with one block fully excluded, and measure the metric on that block. This
is not the same problem as the competition (the block's faults are still *catalogue* faults, and the
model has seen neighbouring terrain), but it is strictly harder than random-window CV and it removes the
adjacency leak that random windows have.

What exists now:

| piece | where | status |
|---|---|---|
| partition (512 px = **51.2 km** blocks, 4 folds, greedy normalised-load assignment, 3 px collar = the metric's R) | `src/blocks.py` | implemented; `tests/test_blocks.py` (22) pins the fold balance that an absolute-mass variant broke ([8192, 8192, 6144, 215552] valid px) |
| training with whole blocks held out | `configs/config_block_holdout.yaml` (`training.holdout: spatial_blocks`) → `src/train.py` | wired; not yet run at scale (`.github/workflows/block-holdout.yml`, `TRAIN_FOLDS`) |
| per-block scoring + paired block bootstrap | `scripts/block_holdout_eval.py`, `src/metrics.block_aggregate`, `bootstrap_from_blocks` | **measured**: proxy DTI 0.0999, CI95 [0.0883, 0.1119], P(width-1 px beats it) = 0.008 |
| fold-restricted scoring (held-out vs trained-on blocks) | `--score-fold K [--complement]` | implemented; the generalisation gap is a pair of measured numbers with CIs, and the report says when a number is only a *reshaping* measurement |
| partition cross-check | `--config` refuses to score on a partition the config does not train against (exit 2) | implemented |

Reported next to catalogue DTI as promised: 0.0999 (new-fault-like) vs 0.2298 (catalogue, in-domain)
for the same artifact, and the two populations agree in sign in 0.747 of blocks on average. **Caveat
carried from `docs/METRIC_STRATEGY.md` §3b:** DTI is not decomposable over blocks, so per-block winners
are not policy winners — blocks buy variance and locality, the global DTI still selects.

**(b) A proxy catalogue for "faults missing from the catalogue".** Compile fault traces for the
GeoDAWN region from sources that are *independent* of the INGENIOUS compilation used for the training
labels — the published state geologic maps (pre-Quaternary faults, which the Quaternary database by
construction excludes), NBMG active-fault products, and published structural studies. Faults present
in the proxy but absent from `labels.tif` behave like the scored population: real faults the training
labels do not contain. The metric on that subset is a *defensible* surrogate for the competition
metric, and it is measurable today. Its weakness must be stated wherever it is reported: the proxy is
built from published maps, whereas the scored faults were drawn by experts looking at the geophysics,
so the proxy can be systematically easier or harder.

*Status 2026-09-18:* the SGMC proxy is built, committed and scored
(`data/evidence/proxy/proxy_catalogue.tif`, 61,664 code-2 px = 6,166 km; a catalogue-copy submission
scores **0.0** there, which is the acceptance control proving the population is not a restatement of the
labels). Its structural weakness is now explicit: **the proxy is only one catalogue**, so it cannot say
whether a policy travels between independently compiled catalogues or merely fits that catalogue's
mapping style. That is item **(b2)** below.

**(b2) A SECOND independent catalogue, and the transfer measurement — MEASURED 2026-09-19, verdict
`REFUSED`: the second catalogue is not independent of the labels.** Emit catalogue A (SGMC), score
against catalogue B (USGS Quaternary Fault and Fold Database, QFaults layer 21 "National Database",
DOI 10.5066/P9BCVRCK, public domain), and report transfer as a prior on hidden-expert-set recall.
The measurement ran on a runner and answered a different question than intended — a more useful one:

* `scripts/fetch_qfaults.py` reads the layer metadata and its renderer vocabulary at run time (nothing
  hand-copied), computes the footprint envelope from `data/labels.tif` (never typed), pages until the
  fetched count **equals** the count the service reports (14,482 verified live 2026-09-18), and writes a
  provenance sidecar with every request URL and live link checks. A silent truncation fails the run —
  a partial catalogue would understate transfer and look like a negative scientific result.
* `scripts/measure_cross_catalogue_transfer.py` runs **controls first** (labels-copy ≈ 0 against B-only,
  B-copy = 1.0, blanket-ones floor), then emits A's code-2 pixels at several widths, scores against B's
  code-2 pixels, and tests `union(model, A)` — a probability maximum, not a mask OR — against a
  **pre-registered** criterion: gain > 0.01 DTI **and** P(union > model) ≥ 0.95 over paired block
  resamples. The verdict is derived: `ADOPT` / `DO NOT ADOPT` / `REFUSED` (controls failed) /
  `NOT MEASURABLE` (no model field).
* Known limits recorded in the code, not discovered later: QFaults is independent of A but **not** of the
  training labels (rules §3.3 — INGENIOUS distributes Quaternary fault layers), so only B's **code-2**
  pixels are scored, and the overlap fraction is measured rather than assumed; if that population is
  empty the measurement refuses to run.

**What the run found.** 14,481 features fetched (`fetched == service_reported`, integrity gate passed)
and rasterised on the competition grid: **169,115** B pixels in total, **108,176** of them outside the
scored footprint (NaN in the submission, therefore unscored), leaving **60,939** inside it. Of those,
**60,938 (100.00 %)** are already within R = 3 px of a training label and exactly **1 pixel** is code 2.
The reciprocal is just as tight: **60,986 of the 60,988** label fault pixels (99.997 %) lie within R of a
QFaults trace. **In this footprint the training labels are QFaults.**

So the population the transfer test needs — faults an expert compiled that the labels lack — is one
pixel wide, and no statistic computed on it means anything. `measure_cross_catalogue_transfer.py`
therefore refuses by pre-registered threshold (`--min-b-only-px 100`, `--min-b-only-fraction 0.005`),
writes `data/evidence/xcat/transfer_report.json` **with the overlap that establishes the refusal**, and
exits 0: a finding, not a failure. A near-empty B raster (< `--min-b-all-px 1000` in-footprint pixels —
empty, misaligned or miscoded) still exits non-zero, so a data bug cannot masquerade as a result.

**What this closes and what it leaves open.** Closed: the idea that a second *Quaternary* catalogue can
supply a hidden-expert-set prior here — none is independent of these labels, and that is now measured
rather than argued. Open: the SGMC proxy population (`data/evidence/proxy/proxy_catalogue.tif`, 61,664
code-2 px, 24.94 % already covered by the labels) remains the only available surrogate for "faults the
labels lack", with its structural weakness intact — pre-Quaternary bedrock structure digitised from
state geologic maps, not an expert interpretation of the GeoDAWN geophysics. Any future candidate
catalogue must be checked for disjointness with `build_proxy_catalogue.py` **before** a transfer number
is quoted.

**(c) Discovery diagnostics on any submission or probability map** — implemented, no extra data
needed (`src/discovery.py`, tests in `tests/test_discovery.py`):

| quantity | meaning |
|---|---|
| `novel_mass` / `novel_fraction` | share of emitted probability mass farther than R=3 px from any catalogued fault. ~0 means the model is restating the catalogue; ~1 with a huge area means it is emitting noise. |
| `candidate_new_faults` | connected components of the thresholded prediction that touch no catalogued fault: the actual candidates Phase 2's expert panel would review. Reports count, area, bounding boxes, longest extent. |
| `catalog_recall_R` | share of catalogued fault pixels with a prediction within R — the mirror image, i.e. how catalogue-driven the model is. |

These are reported automatically by `scripts/blend_submission.py` into every blend report, so every
submission from now on carries its own discovery profile in `data/evidence/runs/<run>/blend_report.json`.

## 4. Experiment queue, in priority order

Each item states its cost and its acceptance criterion *before* it is run, so a negative result is
still a result. Runners are CPU-only (~4 vCPU, 6 h job limit) unless a GPU appears.

1. **Never feed the catalogue to the model as an input band.** Recorded as a rule, because it is the
   obvious "free" feature and it is exactly the failure mode the scored universe punishes. (The
   pipeline does not do this today: `src/dataset.py` reads the 19 feature bands only.)
2. **Label-noise-aware sampling and loss.** Hermant et al. keep only fault-bearing tiles; our config
   keeps 35 % empty windows (`neg_fraction: 0.35`). A/B `neg_fraction ∈ {0.35, 0.1}` and a positive-
   unlabelled weighting (treat unlabelled pixels as *unknown*, not as negatives, e.g. down-weight the
   FP term on pixels far from any catalogue fault while keeping BCE on positives). Acceptance: the
   proxy-catalogue metric (3b) or block hold-out (3a) improves; catalogue DTI may *fall*, which is the
   point.
3. **1 m DEM derivatives as intended feature data.** Rules §2: "the feature data also contain U.S.
   Geological Survey (USGS) Digital Elevation Model (DEM) elevation data at 1-m resolution"; the data
   tab ships links to 716 tiles, already confirmed against the live USGS 3DEP bucket
   (`data/dem_links.json`). Hermant et al. map faults from elevation and slope at 10 m. Plan: fetch a
   subset first (one survey block), derive curvature/roughness/TPI/hillshade at 10–30 m, aggregate to
   the 100 m grid as extra bands, retrain one fold as a pilot. Acceptance: proxy-catalogue improvement
   on the pilot block, then scale to 6 folds.
4. **Spatial block CV everywhere.** Replace random-window hold-out in the ensemble config; keep
   reporting both so the difference is visible.
5. **Calibrate the floor/thinning on the proxy universe** rather than on catalogue labels, and publish
   both tables side by side. This is the step that decides how much mass to emit, and it is currently
   fitted to the wrong distribution.
6. **Lineament post-processing** (Frangi/Hough/segment-linking): keep as an A/B with the same
   acceptance criterion, not as a shipped claim. The About page lists Hough transforms among the
   expected tools; `src/postprocess.py` already implements the vesselness variant and
   `scripts/run_ab_loss_experiment.sh` is the harness.

## 5. What we still cannot know (stated so nobody has to guess)

* The density, geometry and spatial distribution of the scored faults. `|G|` for the test set is
  unknown, so the `0.8·|G|` term in the rearranged metric is unknown.
* Whether the scored faults are concentrated near known fault zones (splays, step-overs) or scattered
  in unmapped terrain. This single unknown decides how damaging it is to emit mass on the known
  catalogue, and only the public leaderboard can answer it.
* The public/private split and its weighting ("the relative weight of faults in both test datasets will
  be determined by the competition organizers before the start of the competition", rules §3.6.2).
* Whether expert review in Phase 2 confirms any given candidate: our candidates are *proposals*, and
  the Phase-2 label set is rebuilt from every team's submissions plus expert judgement.
