## Session 21 additions (2026-09-21)

1. **The leaderboard bar is a moving target, and it moved while we were reading it.** Top of the
   public leaderboard: 0.1972 with 43 ranked (2026-09-16 snapshot) → **0.2854 with 50 ranked**
   (2026-09-21 snapshot, same page, account-free). Both are committed in
   `data/evidence/independent_verification.json` (`competition_standing` +
   `competition_standing_history`). Any sentence that quotes the bar without a date is stale by
   construction; the site renders the bar from the newest snapshot and says when it was read.
   Consequence: the gap between the best local surrogate number for the shipped field and the
   public bar is not a stable quantity, so it is not quoted.
2. **The baseline re-run's `prob_raw.tif` (22.7 MB) is gitignored.** The re-run's audit trail
   (report, `reproduce_audit.json`, `.sha256`, the byte-identical 545,798 B `submission.tif`) is
   committed in `data/evidence/baseline_rerun/`; the raw field is regenerable in one ~5–20 min CPU
   run (`python scripts/baseline_submission.py`), which is exactly what the committed
   `submission.tif` hash match proves. This follows the existing "audit trail in git, rasters
   never" convention; it is noted so nobody assumes the raw field is auditable from a fresh clone
   without re-running.
3. **The CPU route's wall time depends on machine and load.** The committed report's own timings
   total 307.9 s; this session's re-run took ~17 min wall on the same 2 vCPU sandbox while a
   ~4 GB package install competed for CPU. The site now states "≈ 5–20 min (measured 5.1 min on
   2 vCPU)" rather than a single number.

## Session 20 additions (2026-09-20)

1. **The CPU baseline is a floor, not a contender, and it is the only field in this repository whose
   honest number is small.** Generalisation on a fold nothing touched: **combined 0.0788**
   (catalogue 0.0659, proxy-only 0.0709) against the shipped ensemble's committed **0.207431** on the
   same surrogate population. A per-pixel classifier has no spatial context, which is the one thing a
   fault-line detector needs. It exists to make an upload possible from any machine, and its report
   says so in its own `caveats`.
2. **The support cap is pre-registered, not derived.** 5 % of the footprint and ≥ 1,000 px are
   judgement calls made before the search ran, defending against the degenerate *emit everything*
   optimum (measured union DTI 0.1229 at 5,164,312 px). A different cap would select a different
   policy; the report records the cap so the choice is auditable rather than hidden in a default.
3. **Two folds held out instead of one costs real training data.** With the default 4-fold partition
   the model trains on 29,519 positives instead of three quarters of them, so the baseline is
   *weaker* than a single-holdout design would look — that is the price of separating selection from
   measurement, and it is stated rather than traded away quietly.
4. **`scripts/block_holdout_eval.py` still cannot thin**, so a thinned candidate cannot be audited
   from its raw field; the generalisation audit works only because a shipped submission is binary.
   Queued in `SUGGESTIONS.md`.
5. **Link verification cannot be re-run from this sandbox.** 77 of 84 catalog URLs are unreachable
   here, and `scripts/verify_links.py` now refuses to overwrite the measured record instead of
   writing that as a finding. Any change to `docs/data_catalog.csv` therefore needs a networked
   re-verification before its `verified_*` columns can be trusted.
6. **The human gate is still the rate-limiting step.** Enrolment, the first upload, reading the
   private score, the final selection and the generative-AI disclosure are all `HUMAN` in
   `data/evidence/submission_readiness.json`; every machine-doable step up to that click is measured
   and re-verified, and the first upload remains the highest-value next action on this project.

1. **The combined (union) population is a surrogate, not the private truth, and its DTI is not a
   predicted score.** It is the disjoint union of two *local* catalogues — the competition's
   `labels.tif` (60,988 px) and the SGMC proxy's code-2 pixels that the labels do not contain
   (61,664 px) — so it contains **no expert-verified new faults**, which is what rules §3.6 scores
   in both phases. The shipped artifact's **0.207431 CI95 [0.192924, 0.222879]** on that union is
   therefore *not comparable* to the public leaderboard's top score (0.2854 on the latest snapshot)
   quoted as a predicted standing. What it does license: a **third, differently-composed population
   on which the adopted emission policy was re-tested and held** (the only better candidate is
   +0.0054 at P = 0.815, below both pre-registered bars).
2. **`FP_w` is not decomposable across truth populations.** `TP_w` and `FN_w` are sums over *truth*
   pixels and the two populations are disjoint, so they add; `FP_w` is a sum over *prediction*
   pixels (1 − max_g k(d(x,g))), so the union's `FP_w` is **not** the sum of the components'. Any
   sentence of the form "the union DTI follows from the two component DTIs" is false — the union must
   be scored from the rasters, which is why both workflows now pass `--combined-population`.
3. **The four-fold generalisation gap is a spread of point readings, not a distribution with error
   bars.** Folds have 8, 9, 9 and 8 scoreable blocks; a readable block-bootstrap interval needs ≥12,
   so every per-fold interval is flagged `interval_readable: false` by the scorer itself. The gap's
   **sign is not stable across folds** (+0.0114, −0.0070, +0.0139, +0.0093). Per-fold rankings are
   not supported by this evidence, and the **full-grid DTI (0.099859, reproduced digit-for-digit on a
   runner) remains the selection statistic**.
4. **The first pseudo-label fire's raw field is unrecoverable.** Run 35451858126's artifact did not
   carry `outputs/prob_raw.tif`, so that arm can never be scored on the union (or on any future
   population). The re-fire is a **new training run** with the same seed, partition and config, i.e. a
   *replicate*, not the same field: if its proxy-only contrast differs from +0.1036 within training
   noise, that is expected, not a contradiction. The fixed artifact list (`outputs/prob_raw.tif` +
   `outputs/manifest.json`) is what prevents this class of loss going forward.
5. **The baseline arm's union numbers depend on an artifact retention window.** They come from
   downloading run 35413207736's `block-holdout-fold-0` artifact (24.8 MB, `retention-days: 14`,
   **expires 2026-10-03**) and re-scoring it after a sha256 gate. After that date the baseline's union
   population can only be read from the committed sibling JSON this fire produces — or by retraining
   fold 0. The committed JSON survives; the raster does not.
6. **Artifact downloads are egress-blocked in this sandbox**, so the download-and-re-score step
   cannot be exercised locally end to end: it is verified by static lint (the sha256 gate, the
   `run-id` parameter plumbing, the `continue-on-error` path, the missing-file warning) and by the
   API listing that proves the artifact exists and its size. The *scoring* it feeds is the same
   `--combined-population` code path that produced `combined_truth_shipped.json` locally.
7. **The union contrast for fold 0 is measured — and it is under-powered, not decisive.** Run
   35477119490 (2026-09-20T02:06Z) landed both arms on all three populations. On the union, the
   held-out blocks give baseline 0.197183 → pseudo **0.228463**, contrast **+0.031280**,
   **P = 0.916**, CI95 **[−0.010237, +0.082188]** — the interval spans zero, and with only **8
   scoreable blocks** the scorer's own reliability rule marks it COARSE, so P is a spread indicator
   rather than a confidence statement. The trained-on blocks move the *other* way on the same
   population (−0.037531, P = 0.0025, 26 blocks). Derived verdict
   `GAIN_ON_THE_COMBINED_SURROGATE` with `shippable_evidence: false` — and that flag is a **constant
   by construction** in `scripts/read_landed_reports.py`, not a threshold that was evaluated: the
   reader makes no shipping decision. Measured against the bar `docs/FIELD_SELECTION_RULE.md`
   actually pre-registers for adopting a field (R3: paired block bootstrap P ≥ 0.95; R1: a +0.010
   margin), this contrast clears the margin and **misses P**, so it licenses *no* adoption. It is
   suggestive on the only population that contains both fault kinds, and that is all it is.
8. **GitHub Pages is served by the legacy branch build, not by `pages.yml`.** Verified 2026-09-19:
   `build_type: legacy`, `source: {branch: main, path: "/"}`; `/GEMSDOE/docs/executive_summary.html`
   resolves while `/GEMSDOE/executive_summary.html` 404s, even though the Actions deploy job reports
   success with a `docs/`-rooted artifact. **Limitation:** every README/site link is correct for the
   legacy build only; switching Pages to the workflow build would break all of them at once, so the
   mismatch is flagged here rather than "fixed" in one file.
9. **Replicate variability is now measured, and it is the size of the effect.** The two pseudo-label
   fires used the same seed (46), the same config and the same block partition, on different runners:
   fire 1 gave proxy-only **+0.1036** (P = 0.999) and catalogue **−0.0874** (P = 0.001); fire 2 gave
   proxy-only **+0.0677** (P = 0.988) and catalogue **−0.1079** (P = 0.000). That is a ~0.036
   run-to-run swing on one arm and ~0.020 on the other — **the same order as the +0.0313 union
   contrast**, so no single-fold contrast from this pipeline should be read to two decimal places.
   Both fires are in git history, but the second overwrote the first's committed files, so quoting
   either one alone overstates the precision. What would fix it: a contrast pooled over the four
   committed folds (≥12 scoreable blocks per scope) instead of one fold, or repeated fires with the
   spread reported.
10. **Submitting is still human-only.** No DrivenData account or credentials exist in this sandbox
   (the data tab redirects to login), so nothing here can enrol, upload, read a public score, or make
   the single final selection. `docs/submission.html` is the handover checklist for the person who can.

## Session 16 (2026-09-18) — what the new error bars do and do not license

1. **The emission decision now has an error bar, and the bar does not move the decision.** Block-stratified
   scoring of the shipped artifact gives proxy DTI **0.0999 with CI95 [0.0883, 0.1119]** over 56 blocks
   (34 scoreable). The best genuinely different candidate in the recoverable sweep (width 1 px, 0.0878)
   beats the reference with probability **0.008** over paired block resamples. So "keep floor 0.1, width 0"
   is now a statement with a confidence interval behind it, not a point estimate. **Limitation:** the
   interval is over *blocks of one survey*, i.e. it quantifies spatial sampling noise in this footprint —
   it is not an interval over competitions, over truth definitions, or over the hidden expert set.
2. **DTI is not decomposable over blocks, so per-block winners are not policy winners.** TP_w sums over
   truth pixels while the FP penalty is a global mass term; a candidate can win in most blocks and lose
   globally. `block_holdout_eval.py` now derives whether that happened (`interpretation.n_conflicts`, **0**
   in the committed run) instead of asserting it. **Limitation:** blocks are therefore used for *variance*
   and for locating where a score comes from; the **global DTI remains the selection statistic**. Anyone
   tempted to select per block (e.g. widen only in sparse blocks) is outside what this evidence supports.
3. **The floor axis is degenerate for the shipped raster, and that is reported rather than hidden.** The
   artifact is a hard band (two distinct values), so 5 floors × 10 widths produced only **10 distinct
   emissions** — 40 candidates are duplicates detected by sha1 of the emission and pruned from the
   committed per-block rows after every consumer has run. **Limitation:** sweeping floors requires the
   *pre-shaping* ensemble field, which only the runner has (`proxy-eval.yml` re-blends the saved fold
   artifacts); the sandbox cannot re-open the floor question on the shipped bytes.
4. **The field axis was unruled, and one free parameter remains.** At a fixed floor, fields are not
   comparable: floor 0.1 emits 464,736 px on the 6-fold field and 144,738 px on the 16-fold one.
   Comparing at matched support (±15/25/40 % windows, same rule per field) puts the shipped 11-fold mean
   first — 0.0999 vs 0.0850 / 0.0777 / 0.0644 — **stable across all three windows**, so ensemble 1's
   higher 0.1365 was a support effect. **Limitation:** the window width is a free parameter of the
   comparison, which is why all three are reported and the verdict requires stability; and **no
   pre-registered rule covers the field axis** (`data/evidence/emission_field_axis.json` says so). A rule
   must be committed *before* the next re-blend, or the choice stays an unruled judgement call.
5. **Cross-catalogue transfer was measured — and refused, because the two catalogues are the same
   lines.** The runner fetched USGS QFaults layer 21 for the footprint (14,481 features, integrity gate
   `fetched == service_reported` passed) and rasterised it on the competition grid with the same script
   that rasterises catalogue A. The overlap is total: of **60,939** B pixels inside the scored footprint,
   **60,938 (100.00 %)** are already within R = 3 px of a training label and exactly **1 pixel** is
   code 2 (`data/evidence/xcat/qfaults_stats.json`, `transfer_report.json`). The reciprocal is just as
   tight: **60,986 of the 60,988** label fault pixels (99.997 %) lie within R of a QFaults trace.
   **The training labels in this footprint are QFaults.** A DTI computed against a 1-pixel truth
   population would be noise, so `scripts/measure_cross_catalogue_transfer.py` now refuses it by
   pre-registered threshold (`--min-b-only-px 100`, `--min-b-only-fraction 0.005`), writes the refusal
   **with the overlap that establishes it**, and exits 0 — a finding, not a failure. (A genuinely broken
   B raster — under `--min-b-all-px 1000` in-footprint pixels — still exits non-zero, so the two cases
   are not conflated.)
   **Consequence, stated plainly:** the prior on hidden-expert-set recall that EXECUTIVE_SUMMARY §11
   item 1 asks for **cannot be obtained from a second Quaternary catalogue**, because no free one is
   independent of these labels. The SGMC proxy population (`data/evidence/proxy/proxy_catalogue.tif`,
   61,664 code-2 px, 24.94 % already covered by the labels) remains the **only** available surrogate,
   with its known weakness: pre-Quaternary bedrock structure rather than an expert interpretation of the
   geophysics. This was anticipated in the code before the fetch ("if QFaults and the labels turned out
   to be the same lines, that population would be empty and the measurement would refuse to run rather
   than report 0") and is now confirmed with data — **reproduced three times**: two runner fetches and
   one sandbox computation all return 14,481 features, a byte-identical raster (sha256 `3fb2ca73…`) and
   the same REFUSED report with `exit_code: 0`.
6. **Block-holdout *training* is wired but not yet run.** `configs/config_block_holdout.yaml` +
   `--score-fold K [--complement]` can produce a genuine generalisation gap, and the scoring partition is
   cross-checked against the training partition at runtime (disagreement exits 2). **Limitation:** the
   per-block numbers committed today are a **reshaping** measurement on a model that saw every block —
   the report states this in `restriction.note`, and it must not be quoted as transfer to unseen geography
   until a fold trained with `training.holdout: spatial_blocks` is scored on the blocks it never saw.
7. **Unchanged blockers:** human-only (DrivenData account/enrolment, eligibility §1.3, GenAI disclosure,
   final submission selection), infrastructure (no GPU here; 2 vCPU / 3.9 GB RAM), and the private test
   labels. See the table below and `EXECUTIVE_SUMMARY.md` §8d.

---

## Session 15 (2026-09-18) — executive summary polished for submission, 3-pass review, data bridge re-verified

1. **Executive summary ready to submit:** `EXECUTIVE_SUMMARY.md` now opens with a TL;DR 5-command path (`git pull` → `assemble_data_bridge.py` → `prepare_data.py` → `validate_submission.py` → upload) pointing at `data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif` (sha256 `a3dcd6d5…`, 569.5 KB, 11-fold ensemble, floor 0.1/thin/w0, rank 1 of 132). Added **Common Pitfalls** table (6 measured failures), **Data Placement Resolved** proof, **Limitations** table, and **Next Steps** queue — all line-by-line verified. Companion `docs/executive_summary.html` rebuilt with matching TL;DR banner.
2. **Data placement re-verified in sandbox:** `assemble_data_bridge.py` (5 parts → 418,912,844 B, sha256 `4371c82e…`), `prepare_data.py` **PASS** (3292×3730, 19 bands, EPSG:32611, 100 m), `validate_submission.py` on shipped submission **PASS** — evidence and terminal output match. Bridge recovery via `data/bridge/` + runner workflow 35168924460 remains the verified path.
3. **3-pass review completed:** Pass 1 implemented (TL;DR, pitfalls, data proof, limitations, next steps), Pass 2 re-checked every command and fixed hallucinatory verification bullet + duplicate header, Pass 3 re-built site, re-ran `audit_docs.py` **PASS** and `pytest` **175 passed** + `metrics --self-test` 8/8. Site: 10 pages, 0 uncatalogued hosts, 89 catalog rows.
4. **Remaining blockers unchanged:** Human-only (DrivenData account/enrollment, eligibility §1.3, GenAI disclosure, final selection), infrastructure (GPU for full EfficientNet-B5 config), and private test labels — see table #1–10 below and `EXECUTIVE_SUMMARY.md` §8d.

---

## Session 14 (2026-09-17) — executive summary subpage, data placement verified, 3-ensemble sweep confirmed

1. **Executive roadmap delivered:** [`docs/executive_summary.html`](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/executive_summary.html)
   and [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) now provide a complete, verified manual for
   entering and submitting to the competition, including eligibility rules (§1.3), Generative AI disclosure
   narrative template (§3.2), exact GeoTIFF raster parameters, validation commands, and upload procedures.
2. **Data placement completed in sandbox:** Canonical competition rasters in `data/` were assembled
   via `python scripts/assemble_data_bridge.py` and passed `python scripts/prepare_data.py`.
3. **Plain inference path unified:** `src/inference.py` now resolves shaping through `effective_shaping()`,
   adopting the measured winning policy (`floor 0.1, thin, width 0 px`) rather than the in-domain calibration.
4. **16-fold 3-ensemble sweep confirmed:** Sweep on ensembles 1+2+3 (run 35285326679, `eval_sweep-ens123.json`)
   confirmed the adopted policy maintains positive contrast (+0.0581) over reference across 16 live folds.

What remains blocked are the external human-only and infrastructure items below:
- **No DrivenData credentials in automated sandbox:** Submissions require manual web upload by an eligible entrant.
- **Compute constraints:** High-capacity architectures (e.g. SegFormer / EfficientNet-B5 on full 19-band raster) require CUDA GPU compute.
- **Private test set:** The ground truth for new faults is withheld by competition organizers; local proxy numbers serve as relative diagnostics.

## Session 13 (2026-09-17) — which blockers moved

The emission-policy question (a change had to beat the shipped policy by > 0.01 on the
new-fault-like population, dominate it across plausible scored-truth sizes, and reproduce on an
independent ensemble) is now **answered**: `data/evidence/emission_decision.json` passes all three
conditions for floor 0.1 / thin / width 0 px on **three** independently measured fields — ensemble 1
(+0.0954), ensemble 2 (+0.0456) and the shipping field itself, the mean of ensembles 1+2 (+0.0695) —
the cross-ensemble ranking puts that candidate first of 132 by worst-case contrast, and the re-blend
ships it. That closes the *policy* half of row 6 below without touching the *model* half, and it
sharpens row 5 from "the proxy is a stand-in" to a number: the adopted policy is worth +0.0456 on
the worst of the three fields, so the choice is not an artefact of one field.

What is left is detection, and it is now the binding constraint rather than a suspicion: 74 % of the
new-fault-like truth lies more than 12 px from any emitted pixel, a perfect localizer scores 1.0
while the best 16 px band caps at 0.1618, and this model's 0.1365 on that population is 48 % of the
leaderboard top (0.2854, 2026-09-21 snapshot). Rows 1, 2, 7, 8, 9 and 10 below are unchanged.

## Sessions 11–12 (2026-09-17) — current status of every blocker, line by line

Each row states the limitation, what it costs, and the smallest piece of access that would remove it.
Nothing in the repository can substitute for the first row.

| # | Limitation | Measured cost today | What would remove it |
|---|---|---|---|
| 1 | **No DrivenData account or enrolment** | Cannot enter the competition at all: no submission upload (3/week), no public leaderboard, no forum. The data tab redirects to login | A DrivenData account with the GEMS Prize accepted — a person, not code |
| 2 | **No GPU in this workspace** | Training is capped at resnet34 / 45 epochs / ~2.5 h per fold on free CPU runners. `configs/config.yaml` (EfficientNet-B5, 10 folds, 60 epochs) has never been run end to end | One CUDA GPU (A100-class, 24 GB+) or paid runner minutes |
| 3 | **Sandbox egress allowlist** (github.com, api.github.com, codeload.github.com, pypi.org/files.pythonhosted.org) | Dropbox, DrivenData, ArcGIS, ScienceBase, S3 and the Actions artifact host are unreachable from the dev sandbox. Competition rasters arrive as sha256-pinned git objects; every measurement that needs the network runs on a GitHub-hosted runner | Nothing needed — the two-hop design works; a general egress proxy would simplify it |
| 4 | **Hidden scored labels** | Every local number is a diagnostic, never a leaderboard score. The scored faults are the *new* expert set (rules §1.1/§3.3), disjoint from `labels.tif`; a catalogue-copy submission scores exactly 0.0 on the proxy population (asserted) | The public leaderboard, which only row 1 unlocks |
| 5 | **The proxy population is a stand-in** | USGS SGMC state-map traces, not the expert interpretation of GeoDAWN geophysics. It bounds the *choice* between policies, not the score | Nothing available locally; it is a proxy by construction |
| 6 | **Trained on the catalogue, scored on what the catalogue lacks** | Selection (early stopping, pooled floor, fold weights) maximises in-domain DTI, which is the opposite regime: the in-domain optimum is always the narrowest band (0.1903 → 0.0908 at 6 px) | A selection signal on a population resembling the scored one — the proxy is the closest, and it is a proxy |
| 7 | **External data is inventoried but not downloaded** | 1 m DEM derivatives (`src/external_data.py`, `scripts/download_dem_tiles.py`) and the GeoDAWN/INGENIOUS products are unreachable in bulk from here; ~50 GB storage and S3/ArcGIS egress on a runner would be needed | A runner job with storage, or a machine with unrestricted network |
| 8 | **No independent second catalogue exists for this footprint** (measured 2026-09-19) | The transfer measurement ran and refused: QFaults and the training labels overlap by **100.00 %** inside the scored footprint (60,938 of 60,939 B px within R of a label; **1** code-2 px left). Emitting an external fault catalogue therefore cannot be evaluated against QFaults, and the proxy cannot evaluate it either (the proxy IS catalogue A, score 0.0 by construction) | A catalogue that is *demonstrably* disjoint from `labels.tif` — e.g. a future INGENIOUS/GDR expert fault layer for the GeoDAWN footprint, or the organisers' own hidden set. Verify disjointness with `scripts/build_proxy_catalogue.py` before trusting any transfer number |
| 9 | **Eligibility** (rules §1.3: an individual prize competitor must be a U.S. citizen or permanent resident; teams, entities and academia have their own clauses) | Retroactive: ineligible work cannot be submitted regardless of score | Confirmation of eligibility before the deadline |
| 10 | **Deadline artefacts** (rules §3.2: assets sufficient to reproduce + the generative-AI disclosure in the narrative; §3.5: one final submission for both rounds) | Not started; they are a human step on the submission path | Human authoring at submission time |

# Limitations & Required Access — GEMS Prize

**Verified sources:**
- Competition main: https://www.drivendata.org/competitions/306/competition-doe-gems/
- Problem description: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
- Official Rules PDF: https://www.nlr.gov/docs/fy26osti/96647.pdf
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution

## Current Limitations in this Sandbox Environment

### 1. No DrivenData Authentication
- **Issue:** Data download page https://www.drivendata.org/competitions/306/competition-doe-gems/data/ requires login and competition enrollment.
- **Impact:** Cannot automatically download `training_features.tif`, `labels.tif` (or `numeric_features.tif`, `faults.tif`), `sample_submission.tif`, `1m_DEM_links.csv`.
- **Mitigation:** Code in `src/dataset.py` handles both naming conventions and checks multiple candidate paths. `data/README.md` provides manual download instructions. `scripts/prepare_data.py` validates files.
- **Verified:** Attempt to fetch data tab redirects to login page — confirmed via `fetch_page`.

### 2. No GPU / Limited Compute
- **Environment:** Python 3.11.2, CPU-only, no torch pre-installed (verified via `pip show torch` → not found).
- **Impact:** Training large segmentation models (UNet++, DeepLabV3+, SegFormer with EfficientNet-B5) ideally needs GPU (CUDA 12.6/13.0 or Apple MPS). Reference solution recommends GPU: https://github.com/drivendataorg/gems-prize-reference-solution#choosing-gpu-vs-cpu
- **Mitigation:** Code supports CPU, MPS, and CUDA via `torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`. Mixed precision only on CUDA. Config allows reducing batch size, patch size, MC splits for CPU testing.
- **Needed:** 1× A100 24GB+ or equivalent, 5-10h training for full ensemble (MC=10, 50 epochs).

### 3. No Large External Data Pre-downloaded
- **GeoDAWN:** ScienceBase item https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 contains GB-scale zips (22103_area1_grids.zip 42MB, area2_grids 227MB, mag gdb 350MB, spec gdb 378MB, tiffs 43MB + 230MB, plus 3GB+ contractor packages). Cannot bulk-download in sandbox.
- **3DEP 1m DEM:** Many tiles for Nevada region; each tile ~100MB, region needs dozens. Sources: https://apps.nationalmap.gov/downloader/ and https://apps.nationalmap.gov/lidar-explorer/ and AWS https://registry.opendata.aws/usgs-lidar/
- **INGENIOUS:** GDR https://gdr.openei.org/submissions/1391 contains 116MB+ across 9 files (2m probes, earthquake density, paleo features, etc.)
- **Mitigation:** `scripts/download_external.sh` provides verified official links and manual instructions. `src/external_data.py` implements DEM derivative computation (slope, curvature, TPI, TRI, detrended, hillshade) so that once DEM tiles downloaded, features can be generated.
- **Needed:** ~50GB storage, stable internet, possibly AWS CLI.

### 4. Private Test Labels Unavailable
- **Expected:** Private test set is intentionally withheld; only public test subset shown on leaderboard. This is by design per competition structure https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#competition-structure
- **Impact:** Cannot directly optimize for private test; must use cross-validation and public LB.
- **Mitigation:** Implement Monte Carlo CV (70/30 splits) with distance-weighted Tversky metric for validation, as in `src/metrics.py` and `src/train.py`. Target high recall (β=0.8) to generalize to hidden faults.

### 5. No Direct Submission to DrivenData from Sandbox
- **Issue:** Submitting requires DrivenData account and selecting final submission before deadline (3 submissions/week limit per rules https://docs.nlr.gov/docs/fy26osti/96647.pdf section 3.4).
- **Mitigation:** `src/inference.py` generates submission.tif matching exact format requirements (EPSG:32611, 100m, float32 [0,1], same bounds). `scripts/validate_submission.py` (to be added) checks format.

## What We Need Access To for Full Competitive Run

1. **DrivenData Account + Competition Enrollment**
   - URL: https://www.drivendata.org/competitions/306/competition-doe-gems/
   - Required to download training data and submit.
   - Eligibility: U.S. citizen/permanent resident per Official Rules https://www.nlr.gov/docs/fy26osti/96647.pdf section 1.3

2. **Compute Resources**
   - GPU: NVIDIA with CUDA 12.6 or 13.0 (reference solution recommends `uv sync --extra cu126`), or Apple Silicon MPS.
   - CPU: 8+ cores for data loading
   - RAM: 32GB+
   - Storage: 50GB+ for data + outputs

3. **External Data Access (Optional but Recommended)**
   - USGS GeoDAWN: https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7 DOI https://doi.org/10.5066/P93LGLVQ
   - USGS QFaults: https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23 DOI https://doi.org/10.5066/P9BCVRCK
   - USGS 3DEP: https://apps.nationalmap.gov/downloader/ and https://apps.nationalmap.gov/lidar-explorer/
   - INGENIOUS: https://gdr.openei.org/submissions/1391 DOI https://doi.org/10.15121/1881483

4. **Time**
   - Training: 5-10h for full ensemble
   - Inference: ~1h for full region with TTA and ensemble
   - Expert review of all submissions happens after close; winners notified ~60 days after the prize closes (per PDF section 3.6.5); ACH/W-9 within 30 days of notice (A.2)

5. **Documentation for Winners**
   - Per PDF section 3.2 and 3.5: Winners must submit complete code assets + documentation sufficient to reproduce results, consistent with DrivenData's Winning Model Documentation Template.
   - Must indicate generative AI use if applicable (PDF section 3.2)
   - Must sign eligibility certifications

## Allowed External Data — Verified Licenses

Per problem description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#external-datasets and Official Rules PDF section 3.3:

> Participants are allowed to use any additional data sources, provided that the participants possess a license that permits the data to be used in this challenge and shared with the sponsor for evaluation purposes.

Our external data:
- **USGS data (GeoDAWN, QFaults, 3DEP, EarthMRI):** Public Domain (U.S. Government) — https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits
- **INGENIOUS (GDR):** CC License, Publicly accessible — stated on https://gdr.openei.org/submissions/1391 as "Publicly accessible License CC"
- **GBCGE Subsurface:** Public via OSTI — https://www.osti.gov/dataexplorer/biblio/dataset/1987556

All shareable with sponsor.

## No Hallucinations Statement

- All links verified via `web_search` and `fetch_page` on 2026-09-12.
- No synthetic fault data invented.
- File names `training_features.tif`, `1m_DEM_links.csv` from official problem page.
- Reference solution naming drift (`numeric_features.tif` vs `training_features.tif`) flagged as irregularity.

### 1c. Fresh egress matrix — 2026-09-12T22:09Z (post-sandbox-restart re-test)

| Host | curl result |
|---|---|
| github.com / api.github.com | ✅ data flows (git/gh) |
| codeload.github.com | ✅ data flows (200, 1.3 MB tarball verified) — **NEW**, enables any public repo content |
| raw.githubusercontent.com | ❌ exit 35 (TLS data dropped) |
| objects.githubusercontent.com (release assets) | ❌ exit 35 → smp pretrained encoder weights NOT downloadable |
| prd-tnm.s3 / drivendata-public-assets.s3 / www.dropbox.com / huggingface.co | ❌ exit 35 |
| pypi.org / files.pythonhosted.org | ✅ (numpy/scipy/torch/rasterio/smp/geopandas installed) |

**GitHub mirror hunt (definitive negative):** `gh api search/code` on 5 distinctive filenames
(`gems-geodawn-numerical-features`, `existing_faults.tif`, `numeric_features.tif`, `1m_DEM_links`,
`example_submission.tif`) → **0 hits**. The official competition files have no public GitHub mirror.

### 1d. Reconstructed dataset caveat (2026-09-12)

Because the official files cannot enter the sandbox, the pipeline was verified end-to-end on
`data/reconstructed/` — REAL public-source data (GeoDAWN survey 22103 area1 + INGENIOUS QFaults +
earthquake density; see `data/README.md`), **not** the official competition files: different band
set (11 of 16 analog/extra vs official derivations), different processing, subregion extent.
Usable for: pipeline proof, external-data pretraining, sanity CV. Required for leaderboard:
official data-tab files via `scripts/download_competition_data.sh`.

### 1e. Code-vs-docs discrepancies found during E2E (2026-09-12)

- docs claimed torchvision augmentations in the training loop — **not implemented** in `src/train.py` (FaultDataset transform is a no-op placeholder). Docs corrected; augmentation remains a next-step.
  **Resolved 2026-09-13:** label-consistent numpy crop/flips/rot90/noise in `FaultDataset`, regression-tested (`test_augmentation_is_label_consistent`).
- `make_patches` keeps training windows that partially overlap held-out test regions (only fully-covered windows are excluded) → mild CV leakage vs the reference solution's global test-region zeroing. Flagged; fix queued in SUGGESTIONS §.
  **Resolved (verified 2026-09-15 by re-reading `src/dataset.py:make_patches`):** the test mask now zeros both X and y globally *before* train windows are extracted, and windows >25 % covered by test are skipped; regression-tested by `test_patches_have_no_label_leakage`.
- `pretrained: true` configs cannot fetch encoder weights from this sandbox (release-asset host blocked); sandbox runs use `pretrained: false`. On an unrestricted machine `configs/config.yaml` works as-is.
- 3-epoch CPU smoke model ≈ constant-prediction baseline (DTI 0.0519 vs 0.0524) — honest result; not evidence of model quality.

### 1f-2. Sandbox re-test + auth status (2026-09-15, this session)

| Probe (2026-09-15T00:0xZ) | Result |
|---|---|
| github.com / api.github.com (bare GET) | 200 — HTML of the repo/API root via the egress proxy |
| pypi.org / files.pythonhosted.org | ✅ full installs work (torch 2.14.0+cu130 wheel from PyPI installed in `.venv`) |
| download.pytorch.org/whl/cpu | ❌ TLS dropped (use PyPI default wheels instead) |
| raw.githubusercontent.com (direct) | ❌ exit 35 |
| codeload.github.com tarballs | ❌ now returns a 404 stub through the proxy (was 200 on 2026-09-12) |
| www.dropbox.com / drivendata.org | ❌ exit 35 (unchanged) |
| `gh api` reads | ✅ for the first ~30 min, then **"Bad credentials"** — the Arena session `GH_TOKEN` expired mid-session again; `git push` → `Invalid username or token` |
| `fetch_page` (harness-side, not sandbox network) | ✅ reaches drivendata/NLR/GitHub — used this session to verbatim-verify the metric formulas, submission format, rules §1.1/§3.2 and the reference notebook's band table (19/19 match) |

**Impact:** the parallel-ensemble work is committed on the local branch (`arena/01a0a738-gemsdoe`, 2 commits) but cannot be pushed until GitHub is reconnected in Arena; until then the 6-fold workflow cannot be dispatched. No other route exists from this sandbox (verified again above). Owner action: reconnect GitHub → `git push origin arena/01a0a738-gemsdoe` (or re-run this session) and the run starts automatically (the push touches `.github/triggers/ensemble`).

### 1f. Dropbox mirror fetch notes (2026-09-12)
- Dropbox `scl/fi` links carry a short-lived `st` signature; it expired mid-capture of `Digital-elevation-model-links-JSON.pdf` (chunk 1/13 fetched, rest failed). The durable form is `rlkey` + `dl=1` (used in `scripts/download_competition_data.sh`).
- DEM tiles are ~90-380 MB each; the full 3-project list is tens of GB — download on a machine with disk headroom (`scripts/download_dem_tiles.py`).


---

## 2. Fresh-sandbox re-verification, 2026-09-12 (later same day)

The sandbox was reset between sessions (no packages, empty `data/` except `README.md`), so every environment and
pipeline claim was re-established from scratch instead of trusted.

### 2.1 Egress matrix, re-measured 2026-09-12T23:0xZ

| Host | Result |
|---|---|
| `github.com` (incl. `/archive/...tar.gz`), `codeload.github.com`, `api.github.com` | ✅ 200 |
| `pypi.org`, `files.pythonhosted.org` | ✅ 200 (torch 554 MB wheel installed at ~150 MB/s) |
| `www.drivendata.org`, `www.dropbox.com`, `prd-tnm.s3.amazonaws.com`, `www.sciencebase.gov`, `gdr.openei.org`, `pubs.usgs.gov`, `earthquake.usgs.gov`, `services.nationalmap.gov`, `huggingface.co`, `cdn.jsdelivr.net`, `gist.githubusercontent.com`, `raw.githubusercontent.com`, `web.archive.org`, `en.wikipedia.org`, `www.nlr.gov` | ❌ `curl (35) SSL_ERROR_SYSCALL` (TLS dropped by the allowlist proxy) |

Text fetching is still possible through the platform-side `fetch_page` (used to re-read the problem page, the About
page, the DrivenData main page and all 7 chunks of the rules PDF), but **binary GeoTIFFs cannot be delivered that
way**. `raw.githubusercontent.com` is blocked while `codeload.github.com` works — which is exactly why the
public-data reconstruction below succeeded: repo *tarballs* are reachable, file-by-file raw access is not.

### 2.2 What was unblocked autonomously (no manual input)

- **A real dataset inside the sandbox:** `jklinck/geothermal_research` @ `56d78de7a989c12e2dce50cd65a4095df57030d2`
  (tarball sha256 `f78b96a36fd5e814…`), containing USGS GeoDAWN 22103 area-1 grids and INGENIOUS fault/seismicity
  layers. Built with `scripts/build_reconstruction_dataset.py` → 16-band EPSG:32611/100 m stack, 1.42 % fault pixels.
- **Full pipeline executed** on it: train (2 MC splits) → inference → `validate_submission.py` ✅ → metric scoring;
  plus a 20-test suite and `python src/metrics.py --self-test` (8 checks). Details and numbers: `docs/results.html`.
- **Reference solution fully read** (21 notebook cells, cloned) → the baseline hyperparameters cited across this repo
  are now quoted from it rather than paraphrased.

### 2.3 Irregularities found in this review (all flagged, none silently patched)

1. **Cited-but-uncommitted evidence.** Six places referenced `data/dem_links.json` / `data/evidence/…`; `data/*` was
   gitignored (only `README.md` allow-listed), so those artefacts were never in git and vanished with the sandbox.
   Fixed: `.gitignore` now allow-lists `data/dem_links.json`, `data/evidence/**`, `data/reconstructed/provenance.json`;
   `scripts/fetch_dem_links_pdf.py` regenerates them; every doc reference corrected to "regenerate with …".
2. **Dead dependency.** `requirements.txt`/`environment.yml` listed `albumentations`, which the code never used
   (docstrings claimed it). Removed; augmentation is implemented in `src/dataset.py` and is now wired into training.
3. **API bug caught by the A/B run.** `TverskyLoss.forward()` rejected the shared `fp_weight` kwarg, so the
   reference-comparison arm crashed. Fixed (signature unified) — and it is exactly why we run the comparison instead
   of asserting it works.
4. **Docs overstatement removed.** The "Expected gains 0.45-0.75 DTI" table had no measurement behind it; replaced
   by the measured table. Any remaining estimate is labelled as an estimate.
5. **Still true:** the official competition rasters cannot reach this sandbox (login-gated + TLS-blocked), so no
   leaderboard-representative score can be produced here. Run `bash scripts/download_competition_data.sh` on an
   unrestricted machine → `data/`, then `python -m src.train --config configs/config.yaml`.

---

## 3. Post-merge review, 2026-09-13 (after PR #3 / #4 merged)

Merging was followed by reading the **published** pages against the artifacts, which found four
documentation defects and one deployment ambiguity. All are fixed in the repo; the deployment item is
for the repo owner:

1. `docs/results.html` &sect;4 (end-to-end) quoted the &sect;5 A/B arm's shaping numbers and a pre-fix inference
   mass. Fixed, with the correction kept visible; `src/inference.py` now writes
   `outputs_recon/inference_summary.json` so that row is derived from a run record, and `audit_docs.py`
   re-derives it.
2. The old hand-written methodology page (deleted 2026-09-14 when the site was regenerated from measured evidence by `scripts/build_site.py`; superseded by `docs/method.html` and `docs/metric.html`) quoted "perfect line &rarr; 0.91 DTI, shifted 1 px &rarr; 0.61". The code gives
   **1.0000** and **0.6667** (= k(1) = 2/3) and **0.0000** at 3 px; replaced by the measured table and pinned
   by `test_line_geometry_dti_values`. Two pages also described a "7&times;7 max filter" that no longer exists
   (it would credit corners at d&asymp;4.24 px, which the spec's *radius* excludes).
3. 19 `href="../X.md"` links and 11 bare `href="X.md"` links 404 on the site, because Pages publishes
   `docs/` as the site root and does not render markdown. Repointed to the repository on GitHub; the
   audit's new link check enforces it.
4. The audit gate grew from three checks to five (published tables vs artifacts; site link targets),
   each falsification-tested.
5. **Owner action, not code:** `GET /repos/buffedlizard55-lab/GEMSDOE/pages` reports `build_type: legacy`, source `main`, path
   `/`, yet the successful `deploy-pages` job means the artifact (which uploads `docs/`) is what is
   currently live &mdash; `https://buffedlizard55-lab.github.io/GEMSDOE2/` now serves `docs/index.html` and `https://buffedlizard55-lab.github.io/GEMSDOE2/docs/...`
   returns 404. Both builders are therefore racing on every push to `main`. Pick one: set
   **Settings &rarr; Pages &rarr; Source: GitHub Actions** (recommended; the workflow already uploads `docs/`),
   or remove the `deploy` job and let Jekyll build the repo root. Nothing in the repo can settle this
   without the setting change.

## 4. What the deployed-page check caught (2026-09-13, after PR #5)

`docs/index.html` still described the metric as using a "7×7 max filter" — stale text from before the
scorer was rewritten to enumerate the 29 offsets inside the Euclidean radius. It survived because a
mid-session `git checkout -- docs/index.html` (used to undo a *different* bad edit) silently reverted
that paragraph too, and `audit_docs.py` has no way to know a prose sentence is wrong: it checks that
cited paths exist, that tables equal their artifacts, that links resolve and that hosts are catalogued.
Lesson recorded: after any bulk revert, re-read the whole affected page — and prefer reading the
**deployed** page over the local file, which is exactly how this was found. Fixed in the follow-up
commit; the numeric claims in the same area are pinned by `test_line_geometry_dti_values` (20/20 tests).

---

# 2026-09-16 addendum — measured limits, and one that is not a limit

Everything below was re-measured this session with `curl`, `gh` and `rasterio` inside the sandbox;
nothing is inferred from the earlier notes.

## 1. Egress allowlist (re-measured)

| host | result | consequence |
|---|---|---|
| `github.com`, `api.github.com`, `codeload.github.com` | 200 / working | git push, PRs, workflow dispatch, **`api.github.com` artifact *listing*** |
| `pypi.org`, `files.pythonhosted.org` | working | the full dependency set installs (torch included, CPU) |
| `productionresultssa8.blob.core.windows.net` (Actions artifact payloads) | `EOF` | **workflow artifacts cannot be downloaded here**, only *listed* via the API. Measured by attempting `gh run download 35042805806 --name submission-final-35042805806`. |
| `www.dropbox.com`, `drivendata.org`, `s3.amazonaws.com`, `prd-tnm.s3.amazonaws.com`, `sciencebase.gov`, `apps.nationalmap.gov`, `gdr.openei.org`, `community.drivendata.org`, `raw.githubusercontent.com`, `objects.githubusercontent.com` | TLS `EOF` | no direct data download; no release/artifact/raw fetch |

**Not a limit: the competition data is already in hand.** `gems-geodawn-numerical-features.tif`
(418,912,844 B), `existing_faults.tif` (425,830 B), `example_submission.tif` (1,599,597 B),
`GEMS_96647.pdf` (455,140 B) and `Digital-elevation-model-links-JSON.pdf` (23,032,446 B) were
downloaded, hashed and measured by a **GitHub-hosted runner** in session 2/3
(`data/evidence/inventory.json`, `data/evidence/rasters.json`), and a 512×512 window of the real
rasters is committed as `data/fixture/` so the pipeline can be exercised in-sandbox. The runner is the
only machine here with open egress; that is why every data-touching job is a workflow.

## 2. Compute (re-measured)

2 vCPU / 3 GB RAM / 20 GB disk in the sandbox; GitHub-hosted `ubuntu-latest` runners are 4 vCPU CPU-only
with a 6-hour job limit. Consequences actually observed:

* full-suite tests **do** run here (torch CPU wheel from PyPI): 37/37 pass;
* training does **not** meaningfully run here — 3 GB RAM caps patch size and the 2 vCPUs put a 6-fold
  ensemble out of reach;
* the 6-fold ensemble needed **2 h 36 min** per fold on a runner (`run 35042805806`), i.e. a
  leaderboard-grade ensemble on CPU is possible but slow, which is exactly why the fold jobs are
  parallel and time-guarded (`training.max_minutes`).

## 3. Access we still need (human actions — no amount of automation here replaces them)

1. **DrivenData account + competition enrolment.** Required to (a) download the official data-tab
   files under the competition's own terms, (b) *upload* any submission (this is the single blocking
   step between this repository and the leaderboard), and (c) see the public leaderboard, which is the
   only unbiased feedback on the scored universe (3 submissions/week, rules §3.2).
2. **Eligibility** (rules §1.3): US citizen/permanent resident (or a team whose captain is), US
   incorporation for private entities, US-accredited institutions for academics; DOE employees/support
   contractors and FFRDC *institutional* participation are excluded (FFRDC-affiliated individuals may
   compete individually but are not cash-eligible). Worth a check before any prize is contemplated.
3. **A GPU**, to run `configs/config.yaml` (EfficientNet-B5, 10 splits, 60 epochs) rather than the
   CPU-feasible `configs/config_ci_ensemble.yaml`.
4. **Narrative + code-asset submission** at the deadline: the rules require the complete solution
   assets with resource documentation and a generative-AI disclosure; this repository is deliberately
   shaped to be that package, but the upload itself is human.

5. **Proxies are labelled as proxies, but they are still proxies.** Four different quantities appear in
   this repository and none of them is the competition score: (a) held-out DTI against the public
   catalogue (the wrong population — rules §1.1 scores new faults), (b) discovery diagnostics
   (unlabelled), (c) the shift-robustness width curve (a stress test of the writing operator, run on the
   catalogue), and (d) the blanket-ones floor (a constant). The new LOO audit removes one specific bias —
   fitting and scoring the floor on the same folds — and nothing more.
6. **The emission-width stress test was run on one window.** Rows 2048–2560 × cols 1280–1792 of the
   official grid is the only place where a committed submission and the official label raster overlap
   locally. The 6-fold A/B runs on the runner over all six held-out crops, but the absolute numbers in
   `data/evidence/shift_robustness.json` are single-window and are reported as such.

---

## 7. Session-6 re-verification notes (2026-09-16, ~19:5xZ)

- **Pages config re-checked via API:** still `build_type: legacy`, source `main` `/`. The root
  `index.html` redirect remains the live mechanism; the legacy-vs-Actions race from §3.5 is still
  open and still needs the owner setting flip. Not changed by this session.
- **Test count corrected:** §3 says 37/37; the suite had 38 tests on arrival (all passing) and has
  **46** after this session's 8 regression tests. Until the sibling branch's Tests workflow merges,
  the count is enforced only by whoever runs pytest — the workflow will enforce it in CI.
- **A second session is active on `arena/01a0ab54-gemsdoe`** (verify/links/site pipeline, Tests
  workflow — green since run 35142395935). This session reviewed its diff and deliberately left its
  files alone (verify_links, verify_rules_quotes, build_index URL guard, audit count check,
  workflows/tests.yml); the one shared-file edit here (`_scoring_universe`, different hunk) is
  flagged in STATUS §7 with merge instructions.
- **Reblend A/B still running:** run 35133590776 (`blend` job since 18:18Z) executes pre-session-6
  code — its floor/dilate verdicts stand, its `weight_rule_gain` must be disregarded (STATUS §7).
- **Dependency audit:** `requirements.txt` lists `scikit-learn` and `matplotlib`, imported nowhere
  in `src/`/`scripts/`/`tests/` (verified by grep); harmless dead weight, left in place.
  `requirements.verified.txt` remains the install that was actually exercised here.


---

## 5. Session 7 (2026-09-16) — limitations this session ran into, and what they cost

### 5.1 No GPU, and no way to place the competition rasters here

Unchanged and still the single blocker for the width decision's third condition. The sandbox now has
CPU torch 2.14 + smp installed (enough to import and to run the small tests), but a second *ensemble*
that the pre-registered rule requires before the shaping can change needs the 400 MB feature stack
plus hours of GPU. Everything measured this session was therefore measured on **committed evidence
produced by GitHub runners** (fold artifacts, the SGMC proxy raster, the submission rasters), plus
one full local re-scoring run of the committed submission against the committed proxy catalogue
(reproduced exactly: DTI 0.0247, TP_w 1324.08, FP_w 20211.05, on 61,664 px of truth).

### 5.2 The scored population's size |G| is unknown, and the width decision depends on it

This is now quantified rather than hand-waved: the 6-px band overtakes the shipped skeleton only
above 21,328 px (2,133 km) of scored truth, and the constant-ones baseline overtakes it above
22,056 px (2,206 km). Three plausible anchors (GeoDAWN-blocks density ≈26,000 px; the measured proxy
population 61,664 px; the public catalogue 60,988 px) all sit above the crossover — but they are
*assumptions*: the density of expert-interpreted faults inside the flown blocks need not match the
density of state-map fault traces over the whole AOI. A scored truth below ~2,100 km inverts the
conclusion, so the honest state is "conditional", and the decision file says so in those words.

### 5.3 The proxy population is a different population, not a leaderboard

SGMC traces come from state geological mapping published for other purposes; the scored faults are
expert interpretations of GeoDAWN magnetics, radiometrics and lidar DEM. Neither population contains
the other (the catalogue-copy acceptance check proves the proxy is not a restatement of the labels —
it scores exactly 0.0000 on the absent subset). The consequence: proxy DTI values in this repository
are **not** comparable to public leaderboard scores, and a policy that wins on the proxy may still
lose on the scored set. Cross-population *policy* comparisons are the only legitimate use.

### 5.4 Verification of external claims depends on a page-reading tool, not bash

Bash egress in this sandbox is limited to github.com and pypi.org; drivendata.org, sciencebase.gov,
pubs.usgs.gov, doi.org, gdr.openei.org and nlr.gov are all refused. `fetch_page`/`web_search` reach
them and are what the verification page is built from, but anything that requires *downloading a
file* (the competition rasters, the QFFD geodatabase, the DEM tiles) must go through a GitHub runner.
Two claims remain unverifiable here and are listed as such on the verification page: the metric
worked example's PNGs (S3, not reachable) and the data tab itself (login-walled).

### 5.5 Human-only steps still outstanding

DrivenData account + competition enrolment (eligibility: US citizen/permanent resident), the first
submission upload (3/week), the Pages source setting, and — if the project reaches the finalist
stage — the reproducibility assets and generative-AI disclosure required by the rules. None of these
can be done autonomously; each is listed in `SUGGESTIONS.md` §7.1 item 7.

## Session 19 (2026-09-19) — what the third population and the four-fold gap do and do not license


## The browser generator: what it does not cover (added 2026-09-22)

`docs/how_to_submit.html` §3 can now write the submission raster in the reader's own browser, and
`data/evidence/site_generator.json` records that its bytes parse, validate and match the adopted
artifact's float32 pixels exactly. Four limits stay true, and the page states each of them where the
claim is made:

1. **It ships one field, not any field.** `docs/submission_field.bin` is the run-length encoding of
   the *adopted artifact's* prediction (`data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif`),
   and the codec is ternary by construction — 0.0, 1.0 and NaN only. A continuous field is refused
   ("not binary"), not silently quantised, so generating *your* model's output in the browser needs
   the payload rebuilt (`scripts/build_submission_payload.py --tif …`), which is a repository step.
2. **Identical pixels, not identical bytes.** The artifact is 256×256 LZW-tiled; the page writes
   64-row deflate strips (358,332 B vs 570,890 B). Scoring reads pixels, so this is cosmetic — but
   "byte-identical" would be false, and the page does not say it.
3. **No score, ever, from here.** The page cannot know the leaderboard's number: the score exists
   only after a human enrols and uploads (`human_upload`, the one `HUMAN` gate of the 9).
4. **The `.zip` half of the container rule is a transcription.** The dialog's wording — "a
   single-band GeoTIFF (.tif) file, or a .zip file containing a single GeoTIFF" — was read by a
   logged-in human; rules §3.2 documents only the GeoTIFF form. `drivendata.org` is not reachable
   from the sandbox, so this cannot be re-verified here (irregularity 7 on `docs/submission.html`).
   The `.tif` route is the one whose requirement is checkable from the repository alone.

## What could not be verified about the *published* site (added 2026-09-22, session 23)

The browser generator is verified **in this checkout** (the writer runs under node, rasterio agrees
on the pixels, the repository validator passes the file, and route F re-ran all of it on a clean
runner twice). What is *not* verified from here is one hop further out:

* **Serving of `docs/submission_field.bin` by GitHub Pages.** `github.com` is in this sandbox's
  egress allowlist and `*.github.io` is not reachable from `bash` at all, so the only probe available
  was the fetch proxy. It confirms `docs/submission_meta.json` is live (full manifest returned) and
  `docs/style.css` carries the generator styles, i.e. the merged build is the one being served — but a
  532 KB binary was not retrievable through that path, and the first `submission_meta.json` probe
  returned a **cached 404** that a query-string cache-buster disproved. Treat any first `404` on a
  just-pushed Pages asset as unproven. One `curl -I` by a person settles it; the glue's behaviour on
  a failed fetch is a visible "the payload did not load" error, never a silent wrong file.
* **The Pages *build* status, not just its commit.** `gh api repos/…/pages/builds` shows `built` for
  the merge SHA, which is the check to run if the site looks stale after a merge.
* **The submit dialog's wording**, unchanged from before: transcribed by a logged-in human, not
  fetchable here (rules §3.2 documents only the single-GeoTIFF form).
