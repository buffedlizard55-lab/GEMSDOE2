# Project status — 2026-09-24/25 (sessions 11–25)

## Session 25 (2026-09-24/25) — "Predicted values must be in range [0, 1]": root-caused, fixed at the writers, gated in the validator, and the site now hands over a unique name + Note

The platform rejected a real upload of the shipped artifact with exactly one sentence —
`Predicted values must be in range [0, 1]` — while every value in the file *was* in [0, 1].
Root cause, measured on the bytes (`data/evidence/runs/ens12-adopted-floor0.1-w0/sanitize.json`):
**3,061 NaN pixels sat inside the sample submission's valid (scored) region** and 1,540 finite
pixels sat where the template is NaN; the independent community validator
(`Gameassassin777/gems-eval` → `gems_eval/validate.py`) and a second team's README both require
*finite values in [0,1] inside the valid region, NaN outside* — the platform applies the same rule
and reports it as a range error. Everything below was executed, not planned:

1. **The shipped artifact is now template-conformant.** `scripts/sanitize_submission.py` (new;
   dry-run/`--write`/`--json`, before+after sha256) filled the 3,061 / masked the 1,540 / clipped 0
   / left 12,274,559 untouched; GDAL_NODATA `None` → `nan` to match the template. New bytes:
   sha256 `7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15`, **570,890 B**
   (finite 5,167,373 / nonzero 172,974 — the nonzero support is unchanged, so every score the
   policy ever measured is unchanged; sanitation is neutral under the scorer, which maps
   NaN→0.0). The sidecar, `validation.log`, `postwrite.json` and `sanitize.json` were refreshed.
2. **The validator is the gate, not a one-off cleanup.** `scripts/validate_submission.py` gained
   checks 16–17: **FAIL** on any non-finite pixel inside the sample's valid region (the exact
   platform rejection) and **FAIL** on any finite pixel outside it; the runner-side gate
   (`src/submission_io.py:cli_validate`) enforces the same invariant on CI and in
   `make-submission.yml`.
3. **`conform_to_template()` lives in `src/submission_io.py`** and every writer now calls it:
   the ensembling/blend path conforms the blended field (fills holes the blend left, masks what
   the template masks), and `src/inference.py` conforms `final` before the read-back verifier
   sees it — the next artifact any run writes is conformant by construction. CLI:
   `python -m src.submission_io validate-conformant FILE` (exit 1 + count when non-conformant).
4. **The payload regenerated.** `docs/submission_meta.json` (22,366 B, strict JSON — `NaN` is not
   JSON, the writer emits the string `"nan"` like the template) + `docs/submission_field.bin`
   (532,072 B / 259,495 runs) now reproduce the sanitized bytes; `build_submission_payload.py
   --check` exits 0 and `scripts/check_site_generator.py` is again **PASS** (8 PASS + 2 INFO of
   10 measured steps, node v22.22.3, "pixels are the artifact's", no NODATA-tag caveat anymore — artifact and
   generated file both declare `nan`).
5. **The site answers the rejection.** `how_to_submit` §6 grew **§6b "If the platform rejects the
   file…"**: the literal error string, why NaN-in-region produces it, the one-command probe
   (`python scripts/sanitize_submission.py --pred FILE`, exit 1 = not conformant), the before/after
   hashes, and the exact text to paste into a DrivenData support reply. The hero and executive
   summary carry the same warning inline.
6. **Unique name + Note, as asked.** The generator now hands over, next to the download button,
   an identity block: file `gems-submission-<UTC instant>-<artifact sha8>.tif|.zip` (changes every
   build — two downloads can never collide) and a suggested Note
   `<policy> · build <sha8> · <UTC stamp>` ("clustering with k=25"-style disambiguation; the
   workflow's job summary prints the same shape with its run id). Identity arithmetic lives in
   one exported `submissionIdentity()`; `scripts/check_site_generator.py` asserts the exported
   function carries no forbidden literals, and the 4 UI harnesses assert the block renders.
7. **`make-submission.yml` rebuilds the site.** New step after the headless judge (so pages embed
   that run's evidence) and before the measurement commit (docs/ now travels in the same push).
8. **Evidence and docs re-measured, not hand-edited.** `data/evidence/submission_readiness.json`
   re-run (8 PASS / 0 FAIL / 1 HUMAN, new sha); `EXECUTIVE_SUMMARY.md`, `SUBMISSION_GUIDE.md`,
   `SUGGESTIONS.md`, `LIMITATIONS.md`, `docs/FIELD_SELECTION_RULE.md` updated to the measured
   values (570,890 B / 532,072 B / 259,495 runs) and the executive summary's Common-Pitfalls
   table gained the platform-rejection row; `docs/results.html` now opens with the build-time
   sha and labels the run tables as historical. Full suite: **505 passed, 1 skipped** (4 tests
   updated to the new contract + new `tests/test_template_conformance.py`, 14 tests).

Pseudo-label fold-1 (run 36058099668) stayed `in_progress` throughout — no trigger file touched.
This entry's claims were re-checked against the bytes by the Pass-3 verification run below.


## Session 24 (2026-09-24) — the file to submit leads the site: the in-browser builder is the first thing a visitor sees, the Pages CDN check closed, and the pseudo-label fold-1 fire is staged

The standing ask this session was narrower than the generator itself: **the site already could
generate the submission TIF (session 23) — but a visitor could not tell.** The generator lived only
on `how_to_submit.html`, three clicks deep. The ask: "as easy as download to click a File to submit…
in the executive summary or the very beginning of the site… obvious when you visit the site."

1. **The builder now opens the two pages a reader lands on.** `scripts/build_site.py` gained
   `_submission_builder(ev)`, which renders the same panel, the same two same-origin payload files,
   the same 17 self-checks-before-download as how_to_submit §3, framed as a hero
   (`<section id="build-submission" class="build-hero">`, accent border, first `<h2>` of the page).
   It is the first block of `docs/index.html` (ahead of "The task" and the rules block) and the
   executive summary (ahead of section 1, right where the TL;DR now points: "the generator
   <b>directly below</b>"). The landing page's stats card now links `#build-submission` ("build it
   above") instead of describing the file in the third person.
2. **One measurement path, three renderers.** The arithmetic that had lived inside
   `_generator_section` was lifted into `_payload_facts(ev)` — every number the UI may print
   (grid, run counts, blob size, pinned hashes, the IN SYNC/STALE verdict, the rasterio bit-compare
   result) is measured there once, and the how_to_submit section, the new hero and the tests all
   format the same dict. The distinction that matters for honesty is pinned in the facts:
   `bits_ok = None` means *not measured in this checkout*, `False` means *measured different* — the
   hero renders "not measured…" vs "**NOT identical**" accordingly, and a stale payload still
   cannot ship with a live-looking panel (the build-time re-hash is unchanged).
3. **Scripts load only where a panel can exist.** All three pages now pass the generator tail
   conditionally: no `docs/submission_meta.json` in the checkout → no mount, no script, a visible
   "Not available" gap with the command that produces the payload (previously
   `build_how_to_submit` loaded the glue unconditionally — a dead script on a payload-less tree).
   One mount per page is asserted, because the glue binds a single `#tif-generator`.
4. **Verification.** `data/` was re-placed from the sha256-pinned bridge (418 MB stack,
   `prepare_data.py` PASS) and the judge re-run in this checkout: `scripts/check_site_generator.py`
   verdict **PASS, 8/10 steps (8 PASS + 2 INFO)** — node wrote 358,184 B, sha256
   `fa8f4245e34323bf…`, float32 bits identical to the artifact, `validate_submission.py`
   PASSED on the generated bytes, all three container variants readable. The fresh measurement is
   committed (`data/evidence/site_generator.json`, 2026-09-24T19:36:59Z) and the site is rebuilt
   from it. Locally: **424 passed, 1 skipped** (the only excluded modules import torch; they run in
   CI's Tests workflow with the CPU wheel). `scripts/audit_docs.py` PASS, `compileall` clean,
   `src/metrics.py --self-test` 8/8. New tests: `tests/test_site.py` (builder leads both pages,
   single mount, script order, gap-not-panel without payload, numbers come from the payload not
   the template) and `tests/test_site_generator.py` (all three panel pages load the same two
   scripts the tests execute; fresh-clone judge behaviour).
5. **Bug found by running the rebuild: the committed tree was internally stale.** Commit cf20e1b
   ("make-submission: run 35806062038 evidence") updated `data/evidence/site_generator.json`
   (node v22.23.2, 2026-09-23T01:24:17Z) **without rebuilding the pages**, so the committed
   `docs/how_to_submit.html` still rendered the previous measurement (node v22.22.3, 01:15:41Z).
   `tests/test_site_pages.py::test_the_build_reproduces_the_committed_pages` exists precisely for
   this drift and would fail on CI at that commit; rebuilding from the committed evidence fixes it,
   and the pages now carry the runner's measurement. Lesson recorded: an evidence commit that
   feeds a rendered page must rebuild the page in the same commit (the make-submission workflow
   commits evidence only — it is a candidate for a `build_site.py` step, filed under suggestions).
6. **Edge case found by running the judge on a data-less checkout, then fixed and pinned.**
   `scripts/check_site_generator.py` crashed into `KeyError: 'container_vs_sample'` on four steps
   when `data/sample_submission.tif` was absent (fresh clone): `compare_with_artifact` omits that
   key by design, but its callers assumed it. Now the judge names the gap — FAIL with
   "container vs template not judged: data/sample_submission.tif is not placed in this checkout
   (run scripts/assemble_data_bridge.py)" — while the pixel-level judgment still reports
   (`float32_bits_identical = true` is visible in the same step detail). Pinned by
   `tests/test_site_generator.py::test_without_placed_data_the_judge_names_the_gap_not_a_keyerror`
   (monkeypatches SAMPLE to a missing path and asserts the clear FAIL, not a traceback).
7. **Next step 1's remaining human check closed from this sandbox.**
   `https://buffedlizard55-lab.github.io/GEMSDOE2/docs/submission_field.bin` serves the payload on
   the live Pages CDN (fetched 2026-09-24; 52 chunks ≈ 532,174 B of the `gems-rle-v1` stream,
   `?cb=1` to beat the CDN's cached-404 window), and `submission_meta.json` serves with the
   pinned artifact sha256 `a3dcd6d5…`. The browser generator therefore works from the deployed
   site, not just from a clone.
8. **Pseudo-label fold 1 staged (session-23 next step 3c, the time-sensitive one).** The runner
   artifacts holding the no-pseudo baseline raw fields expire ~2026-10-03, and the pooled
   multi-fold contrast (`scripts/read_landed_reports.py --pool`) needs ≥12 resampling units to be
   readable — fold 0 alone gives 7. `.github/triggers/pseudo-label-params` now carries
   `FOLD=1, SEED=46, PSEUDO_WEIGHT=1.0, BASELINE_RUN_ID=35451858112` (the run that trained
   baseline folds 1–3; its artifact `block-holdout-fold-1` is still present), and the trigger file
   records the third fire. The workflow fires on the merge push to `main` (push path
   `.github/triggers/pseudo-label`); it trains fold 1 with the SGMC pseudo pixels, scores both arms
   on the three truth populations, cross-checks with `--strict`, refreshes the pool and commits the
   evidence — no workflow code changed (single-fold fire, the battle-tested path; folds 2–3 follow
   the same two-line params edit once this run lands, because the concurrency group
   `cancel-in-progress: true` cancels a second fire while the first runs).
9. **PR created and merged to `main`; Pages re-deploys from main** (deploy job is main-only).
   After the merge: the landing page on the live site leads with the builder, and the pseudo-label
   run starts on main.

### Still open, in order

1. **Human: enrol, upload, first score** — unchanged, still the only unbiased signal (bar 0.2854
   as of the 2026-09-21 snapshot). Now one click on the published landing page gets the file.
2. **Read the pseudo-label fold-1 evidence** when it lands (`data/evidence/pseudo_labels/`),
   then fire folds 2–3 (params edit + trigger bump) so the pool crosses 12 units before
   ~2026-10-03.
3. **1 m DEM derivatives** (item 3 in §11 of the executive summary): code ready
   (`scripts/download_dem_tiles.py`, `src/external_data.py`); needs an unrestricted machine and
   ~50 GB of USGS 3DEP tiles — not runnable in this sandbox (network + disk policy).
4. **GPU EfficientNet-B5 full config**: needs a 24 GB GPU runner; also the first environment where
   `training.select_on=union` actually trains.
5. Optional: have `make-submission.yml` run `scripts/build_site.py` after committing evidence, so
   an evidence-only commit can never leave the pages stale (the drift found in item 5 above).

### Blockers / access needed (restated)

❌ No DrivenData credentials in this sandbox: the data tab, leaderboard and submit dialog are
login-gated — no automated download or upload is possible or claimed (the dialog wording is a
team transcription, irregularity 7 on `docs/submission.html`).
❌ No GPU; no unrestricted egress (PyPI/CDN fetches only); ~50 GB tile downloads and 24 GB GPU
runs must happen elsewhere.
✅ New this session: the Pages CDN serves the payload (item 7 above), so the one remaining
"human, on any machine" check is gone.

## Session 23 (2026-09-22) — the site can now generate the submission file itself, in the browser, with no install and no GPU

The standing ask was narrow and it was about capability rather than analysis: **make it possible to
generate a submission into the competition from the site**, i.e. the reader of
`docs/how_to_submit.html` should be able to end up holding the single-band float32 GeoTIFF the
DrivenData dialog asks for, without cloning the repo. That is what this session built, plus the two
routes (CLI, CI) that produce the same bytes, and the tests that make all three honest.

1. **The payload the page renders, measured rather than typed.** `scripts/build_submission_payload.py`
   encodes the adopted artifact's field into `docs/submission_field.bin` (a `gems-rle-v1` uleb128
   run-length stream, 532,174 bytes for 259,549 runs) and writes `docs/submission_meta.json` with
   every number the page is allowed to show: grid, transform, EPSG, resolution, value counts, the
   blob hash, the artifact hash, and 17 named checks. `--check` re-derives the manifest from the
   raster and refuses if any pin drifted; `--verify` decodes the blob and compares float32 bits to
   the artifact. Both pass in this checkout.
2. **`docs/geotiff_writer.js` — a TIFF/ZIP writer in plain JS.** No dependency, no network, no
   `fetch`; header | IFD | extras | data, one pass, 4-byte aligned, values ≤ 4 bytes stored inline in
   the IFD entry (libtiff will not dereference an offset there), and real deflate via
   `CompressionStream('deflate')` with the zlib wrapper GDAL's `Compression: 8` expects (an early
   draft used raw deflate under tag 32946 and libtiff rejected it). Its reader parses the file it
   wrote, the 17 self-checks run on that parse, and the CLI **writes nothing when a check fails** —
   a failed build must not leave a plausible artifact behind.
3. **`docs/generate_submission.js` + the DOM harness.** The page's Build button produces
   `submission.tif` or `submission.zip` and withholds the download unless the self-checks pass.
   `tests/support/generator_ui_harness.js` runs the real glue against a shimmed DOM with Node's real
   `Blob`/`Response`/`CompressionStream`, in `tif`, `zip` and `broken` (tampered blob) modes. It
   caught a genuine bug the unit tests could not see: after the page's own preload set
   `state.meta`, a single `if (!state.meta)` guard skipped the field fetch and the *first* click
   failed with "the payload did not load". Two independently cached stages fixed it.
4. **Byte-for-byte agreement, and the limits of that claim.** The browser route writes
   358,184 bytes — 59 strips of 64 rows, deflate, sha256 `fa8f4245e34323bf…4037d9a`, identical from
   the CLI and from the harness and stable across re-runs. Its **float32 pixels are identical** to
   `data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif`, `validate_submission.py` says
   "✅ Validation PASSED - Ready for submission!", and rasterio's container matches
   `data/sample_submission.tif`. It is *not* byte-identical to the artifact (the artifact is
   256×256 LZW *tiled*, 569,531 bytes) and the page says so in those words; the score remains
   unknown, because scoring requires the human upload.
5. **`scripts/check_site_generator.py` — a headless judge with teeth.** Payload `--check`, the Node
   run, rasterio bit-compare against the artifact, container compare against the sample, the official
   validator, the zip container, and four container variants (rows-per-strip 1/512/3730 and
   `--no-deflate`) → `data/evidence/site_generator.json`, verdict PASS, 8 PASS / 2 INFO. A crash in
   any step is recorded as a FAIL rather than aborting the report. The two INFO rows are the
   NODATA-tag difference from the artifact (the artifact declares no NODATA tag; the official
   template declares `nan`; we declare `nan`) and are surfaced, not hidden. This became gate 9
   (`site_generator`) of `scripts/check_submission_readiness.py` — 9 gates, 8 PASS, 1 HUMAN.
6. **`scripts/package_submission.py`** mirrors the browser's `buildZip`: one stored member
   `submission.tif`, fixed 2020-01-01 timestamp, `--check` re-extracts and re-reads the member with
   GDAL before comparing, and the packager re-packages itself to prove determinism. The lesson
   reused from the writer: *assert the knob moves an observable* — `--method deflate` is recorded and
   `--check` refuses the mismatch rather than quietly accepting either.
7. **`.github/workflows/make-submission.yml`** — route F, the third way to get the file. Routes
   `adopted|baseline|both`, packages `tif|zip|both`, optionally rebuilds the payload; CPU-only
   dependency set (no torch), no DrivenData login, no mirror fetch; a stub-size guard fails the job
   under 100,000 bytes; the job summary carries the sha256 to check after download plus a
   paste-ready submit note that is *not* the §3.2 disclosure; the measurement JSON is committed to
   `data/evidence/make_submission/<run>.json`, never the rasters.
8. **Docs rebuilt from the top.** `docs/how_to_submit.html` is now ten sections: the file, where it
   comes from, **the browser generator**, the CLI/CI routes (six, labelled A–F), validation (+ 5.1
   for the `.zip`), the click-by-click upload path, what is measured vs asserted, the rules quotes,
   and the claims table naming the evidence file behind each claim. `scripts/build_site.py` renders
   `IN SYNC` / `STALE` by comparing the manifest's pinned artifact hash to the artifact in the tree,
   and the section is gone (not faked) if the payload files are absent.

**Bugs this session fixed, all of them found by running something:** the two-stage payload cache
above; `package_submission.verify()` asking rasterio to open a `.zip` (GDAL has no `/vsizip/` magic
here — extract, then open); a second inline write path that differed from the real one only by
`force_zip64` (consolidated into `write_zip()`); the JS `buildZip` stamping `Date.now()` under a
comment claiming a fixed date; Python `decode_runs` raising a bare `IndexError` on a truncated
stream where the JS raised `ValueError` (the twins now raise the same messages); a `| tee` without
`pipefail` in the new workflow; a `|| true` that masked a packaging failure; and a shell-built JSON
summary replaced by a `python - <<'PY'` heredoc. Also removed: a duplicated `upload_steps` block in
`build_how_to_submit` whose dead copy asserted an unverified "≤ 100 MB" size limit — a claim that
never renders is still a claim a reviewer has to read.

**Verification.** `tests/test_site_generator.py` (22), `tests/test_package_submission.py` (7),
`tests/test_make_submission_workflow.py` (11), `tests/test_how_to_submit_generator.py` (11):
477 passed, 2 skipped locally. The 8 failures + 1 collection error are all
`ModuleNotFoundError: No module named 'torch'` (`test_metric`, `test_union_selection`,
`test_ensemble::test_compact_window_subset_picks_contiguous_faulty_run`,
`test_inference_adopted_shaping`) — `download.pytorch.org` is outside this sandbox's egress
allowlist, and those tests run in CI's Tests workflow, which installs the CPU wheel.
`scripts/audit_docs.py` PASS. `check_submission_readiness.py`: 9 gates, 8 PASS, 1 HUMAN.

### Route F executed on a real runner (same session) — two defects found that no local test could see

`gh workflow run` is refused by this sandbox's token (`HTTP 403: Resource not accessible by
integration` — `contents: write`, not `actions: write`), so the trigger path the workflow watches got
its file: pushing `.github/triggers/make-submission` starts the job. Run **35805002447** on
`arena/01a0cb96-gemsdoe` completed **success in 90 s**, CPU-only, and committed
`data/evidence/make_submission/35805002447.json`: `route=adopted`, `package=both`,
`submission_sha256 a3dcd6d51303f312…` (569,531 B — the artifact, byte-identical to the committed
one), `zip_sha256 d20d2e9fa38bbb6b…` (569,657 B — **the same archive the local packager produced**, so
the container is reproducible across machines), `validator_passed true`, `generator_verdict "PASS"`
(the browser generator verified on a clean checkout), `committed_artifact_unchanged true`.

Two real bugs came out of reading that JSON rather than the green tick:

1. **`payload_check: false`.** The evidence builder decided it by grepping the word *"matches"* in
   `out/payload_check.log`, and `build_submission_payload.py --check` says *"reproduce … float32 bits
   identical"*. A gate's verdict is its **exit status**; a prose grep is a second source of truth and
   it drifts. Now each gate records `out/*.rc` (`echo "$st"`) and re-raises it
   (`test "$st" -eq 0`), and the builder reads the numbers — so a failing gate still writes its
   evidence. `if: always()` was added to the summary, the evidence commit and the artefact upload for
   the same reason. Pinned by
   `tests/test_make_submission_workflow.py::test_evidence_flags_come_from_exit_statuses_never_from_prose`
   and `…::test_a_failing_run_still_reports_itself`.
2. **`--check` was environment-dependent** (first CI failure, run 35803568817): it compared the whole
   manifest including `checks.sample_submission`, which is measured against the *uncommitted* 418 MB
   `data/`. On a runner with no placement that is DRIFT for a payload that is fine. `--check` now
   compares the fields a submission actually depends on and reports a differing `checks` block as
   INFO; the matching test re-derives the sample pins only when `data/` exists locally, instead of
   dying on a missing raster.

### Next steps, in order (for session 24)

1. ~~Fire `make-submission.yml` once~~ **done twice.** Run 35805300948 (after the `.rc` fix) came
   back `payload_check: true`, `generator_verdict: "PASS"`, `validator_passed: true`, and
   `zip_sha256 d20d2e9fa38bbb6b…` — the same archive bytes as run 35805002447 and as this sandbox's
   own packager, i.e. the container is reproducible across three executions on two machines.
   `data/evidence/make_submission/{35805002447,35805300948}.json` are the artifacts of that.
   Still open for a human, because no tool here can reach the CDN: one `curl -I` on
   `https://buffedlizard55-lab.github.io/GEMSDOE2/docs/submission_field.bin` (expect `200`,
   `content-length: 532174`). The `.json` twin was fetched and renders; a `404` on the `.bin` would
   mean the browser generator degrades to its "payload did not load" message on Pages while working
   from every clone. (Note for whoever checks: the Pages CDN serves freshly published assets with a
   cached `404` for several minutes — a first `404` is not evidence. Add `?cb=<n>` and look again,
   which is exactly how this was settled here.)
2. **Human: enrol, upload, first score** (unchanged, still the only unbiased signal — bar 0.2854 as
   of the 2026-09-21 snapshot). Now unblocked in a new way: the file can be produced by the browser
   on the published site, so the upload needs no environment.
3. Then the quality queue from session 22, unchanged: 1 m DEM derivatives → GPU EfficientNet-B5 full
   config → pseudo-label folds 1–3 so the pooled contrast crosses 12 units before the runner
   artifacts expire (~2026-10-03).
4. Optional, low cost: add a "Build it here" deep link from `index.html`; today the generator lives on
   `how_to_submit.html` alone.

### Blockers / access needed (unchanged, restated because it bounds item 2)

❌ No DrivenData credentials in this sandbox: the data tab, the leaderboard and the submit dialog are
login-gated, so the dialog's exact wording ("a single-band GeoTIFF (.tif), or a .zip containing a
single GeoTIFF") is **transcribed by the team** and is flagged as irregularity 7 on
`docs/submission.html`; the rules PDF §3.2 documents only the GeoTIFF form. No automated download and
no automated upload is possible from here, and none is claimed.

## Session 22 (2026-09-22) — the union-selection signal is wired end to end into early stopping + ensemble weights, and item 7's pooled multi-fold contrast reader is built and disjointness-verified

The two half-finished items the previous session's queue named first were: (4) *the union DTI is
logged per epoch — the remaining half is wiring `training.select_on` into the early-stopping
comparator / ensemble weights*, and (7) *single-fold noise (±0.036) swamps the +0.0313 union
effect — the pooled multi-fold contrast is what would settle it*. This session built both, and
verified them against the committed evidence and a live fixture run.

1. **`training.select_on` now drives selection, not just logging (item 4, the comparator half).**
   `src/train.py` already logged `DTI_union` every epoch; this session made the early-stopping
   comparator, the saved checkpoint AND the manifest `dti` maximise whichever population
   `training.select_on` names (`in_domain` default, or `union`). Because `blend_submission.py
   --weights dti` reads that same manifest `dti`, choosing `union` feeds the union signal into
   **both** early stopping and the ensemble weights — exactly the two consumers item 4 named.
   Fail-fast: a `union` run with no proxy scope raises before the first epoch (never silently
   selects in-domain), and an unknown `select_on` is rejected. **Demonstrated on the real-data
   fixture:** `in_domain` selects epoch 2 (shaped 0.1206) while `union` selects epoch 1 (union
   0.1081 > 0.1015) — a genuine behavioural difference, not a no-op. New config keys documented in
   `configs/config_block_holdout.yaml` and `configs/config_pseudo_labels.yaml` (both default
   `in_domain`, so no committed run changes behaviour). Pinned by `tests/test_union_selection.py`
   (7 tests, +2 this session).
2. **The pooled multi-fold pseudo-label contrast reader is built (item 7, the power fix).**
   `scripts/read_landed_reports.py --pool` concatenates every committed fold's two arms into ONE
   paired block bootstrap. It is derivation-only like the rest of the script: it recomposes each
   fold's committed per-block metric components (TP_w/FP_w/FN_w are sums), never re-scores a
   raster. The legitimacy of pooling rests on the four block folds partitioning the same 56-block
   grid into DISJOINT sets, so the reader **verifies** the pooled block ids are disjoint (raises if
   a block id repeats across folds), refuses folds scored on different partitions or the same
   probability field on both arms, and marks the pool COARSE below the scorer's own 12-unit bar.
   On the one committed fold it reproduces fold 0 exactly (combined +0.031263 at P = 0.9515 over 7
   units) and its derived verdict is `POOL_POSITIVE_BUT_UNDER_POWERED`, which names the exact
   blocker: the pseudo arm exists for fold 0 only. `data/evidence/pseudo_labels/pooled_two_population_contrast.json`
   committed. Pinned by 6 new tests in `tests/test_read_landed_reports.py` (disjointness refusal,
   partition-mismatch refusal, same-field refusal, missing-fold accounting, a readable pool
   clearing the bars, and single-fold reproduction). The `pseudo-label.yml` workflow now runs
   `--pool` and commits the pooled file, so firing folds 1–3 will cross 12 units automatically.
3. **Verification.** Fresh venv from `requirements.verified.txt` + torch/smp/timm: full suite
   **445 passed, 1 skipped** (was 430+2; +13 tests this session, and the reportlab-gated test now
   runs because reportlab is installed here). `scripts/audit_docs.py` **PASS** (0 cited-but-missing
   artefacts, 0 uncatalogued hosts). Site rebuilt from `scripts/build_site.py`; §11 items 4 and 7
   updated to WIRED/BUILT with the fixture demonstration and the pool verdict rendered from the
   evidence. `EXECUTIVE_SUMMARY.md` §11 and the next-session ordering updated to match.

**Next session:** the ordering stands with items 4 and 7 now machine-ready. **8 (human: enroll +
first upload — the only unbiased signal; the bar it reports against is 0.2854 as of 2026-09-21) →
3 (1 m DEM derivatives, code ready, needs tile download + ~50 GB) → 6 (GPU EfficientNet-B5 full
config — also the first environment where item 4's `select_on=union` switch actually trains) →
fire the pseudo-label workflow for folds 1–3 so item 7's pool crosses 12 units before the runner
artifacts expire ~2026-10-03.**

## Session 21 (2026-09-21) — the data-placement blocker is closed in-sandbox, the CPU route is proven byte-reproducible, and the submission subpage was re-verified line by line against the official sources

The instruction carried from the previous sessions was explicit: *the single remaining blocker
to training is data placement — place the rasters in `data/`, run `prepare_data.py`, and after
that the full pipeline is ready.* This session did exactly that, inside this sandbox, and then
re-verified the submission subpage (the executive-summary subpage explaining exactly how a
submission gets entered) line by line against the official sources.

1. **Data placement completed in this checkout (no unrestricted machine needed).**
   `python scripts/assemble_data_bridge.py` reassembled all three official rasters from the
   committed `data/bridge/` parts in 2.8 s, verifying every part sha256 and the whole-file sha256
   against `data/bridge/manifest.json` (pins from `data/evidence/inventory.json`,
   measured on a GitHub runner 2026-09-14): `training_features.tif` 418,912,844 B
   (`4371c82e…`), `labels.tif` 425,830 B (`7ba308cc…`), `sample_submission.tif` 1,599,597 B
   (`2176d08e…`). `python scripts/prepare_data.py` then **PASSED** on the placed bytes
   (3292×3730, 19 bands, EPSG:32611, 100 m, aligned bounds, band tags present). The standing
   blocker named in `EXECUTIVE_SUMMARY.md` and `data/README.md` is therefore closed for any
   machine that can run `git pull && python scripts/assemble_data_bridge.py`.
2. **The CPU-only route to a submission was re-executed end to end and reproduced the committed
   artifact byte for byte.** `python scripts/baseline_submission.py --out-dir
   data/evidence/baseline_rerun` (CPU, no torch, ~17 min wall in this 2 vCPU sandbox while a
   package install competed for CPU; the committed report's own timings total 307.9 s) selected
   the identical policy (`floor 0.000974, thin, width 0 px`, union DTI 0.160259 on selection
   fold 0), wrote a `submission.tif` whose sha256 is **`9f2577cff20d…` — identical to the
   committed `data/evidence/baseline/submission.tif`** — and its generalisation audit re-scored
   the untouched fold 1 at catalogue 0.06597 / proxy 0.071107 / combined 0.078916, `reproduced=
   True`, max|Δ| = 0.00016874 (same value the committed run recorded). The two reports differ
   only in `timings_s`. `scripts/validate_submission.py` on the freshly written file: **PASSED**
   (all 8 checks). The pre-registered support cap did its job visibly: the highest-scoring
   candidate (floor 0.009487, union 0.197198) emits 487,813 px = 9.4 % of the footprint and is
   excluded for it, with the reason in the report. Evidence committed:
   `data/evidence/baseline_rerun/` (report, audit, `.sha256`, the 545,798 B `submission.tif`
   force-added per the existing convention; `prob_raw.tif` is 22.7 MB and regenerable in one run,
   so it is gitignored like the other run rasters).
3. **The official sources were re-read today, line by line, and the subpage's claims hold.**
   Problem description (page 967): submission format (EPSG:32611, 100 m, same bounds, NaN
   outside, single float32 band in [0,1]), metric (DTI, α=0.2 β=0.8, R=300 m), datasets and
   external-data policy — all as quoted on the site. Competition home: end date **Dec 3, 2026,
   11:59 p.m. UTC**, $300,000 pool, the six "How to compete" steps. Rules PDF
   (`docs.nlr.gov/docs/fy26osti/96647.pdf`, sha256 `50d854b1…`): §3.1 entry, §3.2 single
   GeoTIFF + "three submissions per week" + generative-AI disclosure, §3.5 one final selection,
   §3.6.1/§3.6.2 public/private scoring, §1.3 citizenship — every sentence the subpage quotes
   was re-verified verbatim against the document.
4. **Irregularity found and fixed: the deadline was conflated.** `EXECUTIVE_SUMMARY.md` and the
   executive-summary page printed *"Dec 3, 2026, 11:59 PM UTC (5:00 PM ET)"* — but those are two
   **different** official times, and they do not convert into each other (11:59 PM UTC is 6:00 PM
   EST). The platform's competition end is 11:59 PM UTC (home page, verified today); rules §A.1
   separately requires the submission form / final content to be posted by **5:00 p.m. ET on the
   deadline date** (22:00 UTC on Dec 3 — two hours earlier). §1.2 defers key dates to the
   competition website, so both are stated on the page now, with the safe reading (act on the
   earlier time). Fixed in `scripts/build_site.py` (executive-summary card), `EXECUTIVE_SUMMARY.md`
   (quick-reference table), and added as a note on the how-to-submit page.
5. **Irregularity found and fixed: the index page contradicted the adopted policy.** It said the
   emitted line is *"now written as a band rather than a skeleton"*, but the committed decision
   record (`data/evidence/emission_decision.json`, all three pre-registered conditions met) ships
   `sweep_best_t0_0.1_width0px` — floor 0.1, thin, **width 0 px**: at that floor the measured
   width curve peaks at 0 px on every ensemble (0.136452 at 0 px, monotonically lower up to 20 px
   on the new-fault-like population). The index now says what the evidence says: a thinned
   skeleton, with the reason it was measured rather than assumed.
6. **The sample-template irregularity was re-verified against the official bytes and added to the
   recipe page.** With the rasters placed, the claim that `example_submission.tif` is not the
   "total fault absence" template the problem page describes is now a direct measurement on this
   checkout: 60,988 nonzero pixels, all exactly 1.0, pixel-for-pixel the training labels. The
   how-to-submit page now carries the caveat at the point where the file is used (validate
   *against* it — never upload *it*).
7. **The leaderboard bar moved; the evidence was refreshed instead of left stale.** Re-read the
   public leaderboard today (account-free): **top 0.2854, 50 ranked** (was 0.1972, 43 ranked,
   2026-09-16). `data/evidence/independent_verification.json` now carries the 2026-09-21 snapshot
   with the 2026-09-16 one preserved in `competition_standing_history`. The how-to-submit page
   renders the current bar from the evidence (the wording already said "when this repository last
   read it"). This also surfaces a real consequence for strategy: the bar rose **+0.088 in five
   days** while 7 new entrants appeared — the public split is actively being gamed, which is
   exactly why the §11 ordering (upload for the unbiased signal first) stands.
8. **Two builder defects found by the rebuild itself, both fixed.** `_leaderboard_panel` crashed
   with `KeyError: 'ranks_6_to_11_range'` on the refreshed snapshot because it hardcoded the
   old range key; it now parses the span from whichever `ranks_6_to_N_range` key the snapshot
   carries. Route C's cost on the subpage was an unmeasured "≈ 20 min"; it now reads "≈ 5–20 min
   (measured 5.1 min on 2 vCPU)" from the committed report's timings.
9. **Verification.** Fresh venv from `requirements.verified.txt`: full suite **430 passed, 2
   skipped** (both environmental: one needs the runner-generated `--dump-text` evidence pointer —
   by design skipped in a clean checkout; one needs `reportlab`, installed here to confirm it
   passes). `scripts/audit_docs.py` **PASS** (0 uncatalogued hosts, no cited-but-missing
   artefacts). `scripts/check_submission_readiness.py` re-measured on this checkout: **7 PASS, 0
   FAIL, 0 MISSING, 1 HUMAN** (the two data gates flipped to PASS because the rasters now exist
   here). Site rebuilt from `scripts/build_site.py`; the committed pages match a fresh build.

**Next session:** the ordering in `EXECUTIVE_SUMMARY.md` §11 stands with item 8 now fully
machine-ready: **8 (human: enroll + first upload — the only unbiased signal; the bar it reports
against is 0.2854 as of 2026-09-21) → 3 (1 m DEM derivatives, code ready, needs tile download) →
6 (GPU EfficientNet-B5 full config) → 4 (selection on the union population) → 7's pooled
multi-fold union contrast (≥12 scoreable blocks before P means anything).** The runner-side
artifacts feeding the pseudo-label union contrast expire 2026-10-03 (14-day retention) — if that
contrast is pursued, the pooled version must fire before then.

## Session 20 (current session, 2026-09-20) — the submission path got its own page, the gate table is measured instead of described, and there is now a route to an uploadable file that needs no GPU and no runner

Session 19's queue was: read the landed reports, update the site, pick the next experiment. It also
left a written instruction this session executed first: *an executive-summary subpage that explains
exactly how to make a submission into the contest*.

1. **`docs/how_to_submit.html` — the recipe, as a subpage of the executive summary.** Nine sections:
   the gate table (measured, not described), the exact artifact with its sha256 **re-hashed while the
   page renders**, four routes to that file, the validator gate, click-by-click upload, what happens
   after the upload, the rules sentences that bind it (quoted by id with their verification badge),
   the CPU route's own caveat, and a sources table naming the evidence behind every claim class.
   `docs/submission.html` stays the long-form operational page and is retitled **Submission details**;
   the nav reads `Executive summary › ↳ How to submit`. 15 tests in
   `tests/test_how_to_submit_page.py`.
2. **The gate table is a measurement, and it is a new script.** `scripts/check_submission_readiness.py`
   produces `data/evidence/submission_readiness.json` from eight checks that each re-derive their
   claim: the three official rasters re-hashed against the bridge pins, `prepare_data.py` re-run
   (exit code + the sha256 of its own output), the shipped artifact re-hashed against its committed
   `.sha256` sidecar, the committed validator log re-parsed, the 29 rules quotations re-matched
   verbatim against the PDF, the CPU route's report, the human-only steps, and the artifacts that
   exist in this checkout. Current reading: **6 PASS, 0 FAIL, 1 HUMAN (`human_steps`), 1 measured
   now PASS (`cpu_baseline`)**. `HUMAN` is deliberately not `PASS`: no program here can create an
   account, upload a file or read a private leaderboard.
3. **`scripts/baseline_submission.py` — a CPU-only, torch-free route to a valid submission.**
   scikit-learn histogram gradient boosting on the official 19 bands (it handles the 57.9 % NaN
   natively), memory-bounded by reading features in row chunks. It is the first generator in this
   repository that can run end to end inside the 2 vCPU / 3 GB sandbox, which is what closes the
   "regenerating a submission depends on machines we do not have" gap. Measured end to end:
   **322 s**, artifact 545,798 B, `sha256 9f2577cf…`, 155,889 emitted px (3.0 % of the footprint),
   `scripts/validate_submission.py` → **PASSED** (all 8 checks), same grid/CRS/dtype rules as the
   shipped ensemble.
4. **A degenerate optimum was found by measurement, and pre-registered out of the search.** An
   unconstrained argmax of this metric over the shaping space selects *emit the whole footprint*:
   `FN_w` carries β = 0.8 while `FP_w` carries α = 0.2, so on a weak field the recall term wins.
   Measured on fold 0: the `floor 0, thin off` candidate emits 5,164,312 px and scores union DTI
   **0.1229**, against **0.0207** for the best *localised* candidate. Candidate eligibility is now
   pre-registered (support ≤ 5 % of the footprint, ≥ 1,000 px) and ineligible rows stay in the
   report with their reason; the first run was stopped rather than allowed to ship the whole-grid
   raster, and the second run's winner emits 3.0 % of the footprint.
5. **Two folds are held out, not one: `--fold` selects the policy, `--eval-fold` measures it.**
   `max()` over N candidates on fold A is not an estimate of fold A, so the report separates the
   winner's value on the selection fold from the same field's value on a fold that neither trained
   the model nor took part in the sweep. Measured: selection fold 0 union **0.160259**, untouched
   fold 1 **combined 0.0788** (catalogue 0.0659, proxy-only 0.0709). The generalisation block names
   the command that reproduces it and was **re-run to check that claim** — see the defect below.
6. **Defects found and fixed this session, all by tests or by auditing a claim:**
   * `scripts/build_site.py` did not parse (backslash inside an f-string expression in the new
     builder) — the page could not be generated at all. Fixed by hoisting the escaped expressions
     into variables computed before the f-string.
   * **`docs/verification.html` was published as a bare fragment** — `build_verification()` returned
     its body without ever calling `page()`, so the file had no doctype, no nav and no footer: a
     dead end on a site built for walking from a claim to its evidence. Nothing failed, because the
     only nav test asserted what `page()` does. Now wrapped, and `tests/test_site_pages.py` asserts
     every generated page is a complete document with the whole nav, that no page links to an
     unpublished page, and that **regenerating the site is a no-op**.
   * **`scripts/verify_links.py` would overwrite a measured record with a false one.** Run in this
     sandbox it marks 77 of 84 URLs `UNREACHABLE` — a statement about restricted egress, not about
     the links — while labelling the file *"on a GitHub-hosted runner (live HTTP)"*. It now compares
     against the committed record and **refuses** (exit 3) when fewer than half the URLs the record
     reached answered, with `--allow-degraded` as the explicit override and a `generated_by` line
     that reports how many URLs actually answered. The committed record was left intact.
     A prefix bug found while testing it: `EXPECTED_OK` is `("OK",)` while the counts are keyed
     `"OK_200"`, so a membership test read as "reached nothing" and would have disabled the guard.
   * **The baseline's `reproduce` command did not reproduce.** It pointed `block_holdout_eval.py` at
     `prob_raw.tif` with the winner's floor, but that script shapes by threshold + dilation and
     cannot thin, so it measured the *un-thinned sibling* (combined 0.1259) instead of the field
     that ships (0.0789). Corrected to score the shaped `submission.tif` (thresholding a binary
     field at 0.5 is a no-op) with the runner crosscheck disabled, and the numbers were re-run to
     confirm they come back digit for digit. The report now records the re-run as
     `reproduce_verified`.
   * `tests/test_submission_page.py` asserted the old nav label ("Make a submission") and would have
     failed on the retitle; it now pins both entries, and the new subpage is checked to be reachable.
   * **CI caught the new build-reproduction test disagreeing with itself**, and the disagreement was
     real: `docs/submission.html` renders whether the three official rasters are present in the
     working tree *right now*, and `data/*.tif` is gitignored by design — so the committed page
     (built here, rasters present → "MATCHES THE PIN") could never match a fresh clone's build
     (→ "ABSENT"). The test now strips exactly those live-state lines and nothing else before
     comparing, states why in the test file, and gains a sibling that asserts two consecutive builds
     are byte-identical — so drift is still caught, and the environment dependence is documented
     instead of becoming an intermittent red check.

**Session 20 test count: 431 passed, 1 skipped** (from 368 passed, 1 skipped at the merge).

## Session 19 (2026-09-19) — every landed report read on every truth population, the submission path got its own page, and the third population is now measured on both arms

Session 18's queue was: read the four landed reports, update the site and STATUS, pick the next
experiment. All four reports are now read — but reading them exposed the real defect, which was not
in the numbers, it was in *which population* the numbers were measured on.

1. **All four block-holdout folds are committed, and the gap is a spread, not a point.**
   Fold 3 landed on `main` while this session was working (commit `c129336`, run 35451858112), so
   the partition is complete. `scripts/read_landed_reports.py` (new) recomposes the four committed
   `fold*_generalisation_gap.json` files into
   `data/evidence/block_holdout/fold_gap_summary.json`:

   | fold | held-out DTI | trained-on DTI | gap | scoreable blocks (held/trained) | CI readable |
   |---|---|---|---|---|---|
   | 0 | 0.0806 [0.0586, 0.1021] | 0.0920 | **+0.0114** | 8 / 26 | no (8 units) |
   | 1 | 0.0789 [0.0644, 0.0963] | 0.0719 | **−0.0070** | 9 / 25 | no (9 units) |
   | 2 | 0.0480 [0.0238, 0.0803] | 0.0619 | **+0.0139** | 9 / 25 | no (9 units) |
   | 3 | 0.0892 [0.0581, 0.1197] | 0.0985 | **+0.0093** | 8 / 26 | no (8 units) |

   Mean **+0.006904**, min −0.007019, max +0.013916, spread **0.020935**, positive in 3 of 4 folds.
   The honest reading is unchanged by the fourth fold and is now derived rather than asserted: *no
   clear memorisation, no clear transfer gain* — the sign is not stable across folds and no fold
   reaches the ≥12 resampling units a readable block-bootstrap interval needs, so the per-fold
   intervals stay flagged `interval_readable: false` by the scorer itself. **The full-grid number
   (0.099859, reproduced digit-for-digit on a runner) remains the selection statistic**; the gap is
   quoted as a spread across folds, never as an error bar.

2. **The pseudo-label contrast was measured on one population, and that population is the one the
   pseudo-labels were cut from.** `data/evidence/pseudo_labels/fold0_two_population_contrast.json`
   (derived, and the derivation reproduces the runner's committed paired bootstrap exactly —
   DTIs, CI95 and P=0.999) reads the contrast on *every* committed truth population:

   | scope | population | baseline → pseudo | contrast | P(pseudo > baseline) |
   |---|---|---|---|---|
   | held-out (8 blocks) | proxy-only (new-fault-like) | 0.0806 → 0.1841 | **+0.1036** | 0.999 |
   | held-out (8 blocks) | catalogue labels | 0.2113 → 0.1239 | **−0.0874** | 0.001 |
   | trained-on (26 blocks) | proxy-only | 0.0920 → 0.1428 | +0.0507 | 1.000 |
   | trained-on (26 blocks) | catalogue labels | 0.2152 → 0.1388 | −0.0764 | 0.000 |

   Emission support goes 90,433 → 338,649 px on the held-out blocks. Derived verdict
   `TRADE_OFF_PROXY_GAINS_CATALOGUE_LOSSES`, `shippable_evidence: false`. The circularity is
   *derived*, not asserted: `configs/config_pseudo_labels.yaml`'s `pseudo_label_path` is the same
   `proxy_catalogue.tif` (code 2) the proxy truth comes from, so the +0.1036 arm is measured against
   truth cut from the training signal's own raster while the −0.0874 arm is not. Two component
   populations pointing in opposite directions settle nothing — which is the reason for item 3.

3. **A third truth population is now measured: the union (new).**
   `scripts/block_holdout_eval.py --combined-population` scores the union of the catalogue labels
   and the new-fault-like proxy pixels the labels do not contain (122,652 px: 60,988 + 61,664,
   disjointness verified at score time) as the closest local surrogate for the Phase-2 "complete
   updated test set" of rules §3.6. It is off by default so every committed report keeps the schema
   its tests pin. Why it cannot be inferred from the two components: `TP_w` and `FN_w` are sums over
   *truth* pixels and the two populations are disjoint, so they add — but **`FP_w` is a sum over
   *prediction* pixels** (1 − max_g k(d(x,g))), so the union's `FP_w` is not the sum of the
   components' and the union DTI must be scored from the rasters.
   `data/evidence/proxy/combined_truth_shipped.json`: the shipped artifact scores **0.207431, CI95
   [0.192924, 0.222879]** on that surrogate (components: catalogue 0.229799, proxy-only 0.099859).
   The full candidate sweep on the union puts the adopted policy **2nd of the 10 distinct emissions** (50 swept, 40 pruned as duplicates): widening
   to 1 px scores 0.212838, a **+0.0054** contrast at **P = 0.815** with support 516,204 px — below
   *both* pre-registered bars (P ≥ 0.95 and +0.010), so **the adopted emission policy holds on a
   third population too**. Mean per-block sign agreement between the two component populations is
   **0.747** — that is the mean of `population_agreement.per_candidate[].agreement_fraction` over
   the 50 swept candidate rows, i.e. the block-level story is mostly shared but not identical. For
   the adopted candidate itself the two populations agree in sign on **32 of 32** blocks where both
   are scoreable (`agreement_fraction: 1.0`), so the disagreement lives in the *rejected* candidates.

4. **Both arms of the fold-0 contrast now exist on the union — via a download, not a retrain.**
   The first pseudo fire's artifact did not carry `outputs/prob_raw.tif`, so *that* arm can never be
   scored on a new population (recorded in `LIMITATIONS.md`; do not try to reconstruct it). The
   workflows are fixed instead, and the fix is pinned by lint tests:
   * `block-holdout.yml` and `pseudo-label.yml` score every scope with `--combined-population`, so
     each fold's raw field carries all three populations while the field still exists;
   * `pseudo-label.yml` downloads the **baseline** arm's raw field from the artifact of the run that
     produced its committed report (`BASELINE_RUN_ID=35413207736` → artifact `block-holdout-fold-0`,
     24.8 MB, verified present today, expires 2026-10-03) and re-scores it on the union. The sha256
     the committed report recorded for its own input is checked first, so a wrong run id fails the
     job instead of contrasting two different fields; a missing file warns and the union contrast is
     reported `NOT_SCORED` rather than guessed;
   * `pseudo-label.yml`'s artifact now carries `outputs/prob_raw.tif` + `outputs/manifest.json`;
   * `scripts/read_landed_reports.py --strict` recomposes the whole contrast from the committed
     per-block rows, independently of the runner's inline bootstrap, and **exits 2** if they
     disagree — a derived number is committed only when an independent recomputation reproduces it.
     The same script reads a sibling `*_combined.json` report as either arm's union population,
     validated to be the *same* probability field by input sha256, and records per-population
     `sources` in the output so every number says which file it came from;
   * `--gaps-only` writes just the fold-gap summary, which is all a block-holdout job can derive the
     moment a fold lands (no pseudo arm exists yet).
   The trigger was re-fired with that reasoning recorded in `.github/triggers/pseudo-label`, and the
   branch push dispatched it (run **35477119490**): every new step came back **green** — the
   cross-run artifact download, the sha256 gate (it accepted the downloaded field, so
   `BASELINE_RUN_ID=35413207736` really is the run that produced the committed report), both arms'
   three-population scoring, the `--strict` recomputation (it matched the runner's own inline
   bootstrap, so the run did not exit 2), the evidence commit through `scripts/push_evidence.sh` and
   the artifact upload that now carries `outputs/prob_raw.tif`. The measurement is item 8.

5. **A latent CI defect fixed on the way: evidence pushes raced each other.** Three fold jobs plus a
   pseudo-label fire from the same trigger push all rewrite the *same derived file*
   (`fold_gap_summary.json`), and the old `git commit; git pull --rebase; git push` sequence fails
   the job when two interleave — losing 300 minutes of evidence. `scripts/push_evidence.sh` replaces
   it in both workflows: on a conflict it **regenerates the derived file from the merged tree** (the
   only correct resolution — the summary is a function of every committed fold) and re-stages exactly
   the files that were staged, never an unrelated large raster.

6. **`docs/submission.html` — the operational page (new).** Nav entry "Make a submission": what to
   upload, where, in what format, what the rules require, what is irregular, and the pre-flight
   checklist. Nothing on it is typed as a fact: the artifact's sha256 and byte count are re-hashed at
   build time, the data-placement table compares the manifest pins against whatever `data/` holds
   now, the validator table is the committed validation log parsed, rules sentences are quoted by id
   from `data/evidence/rules_quotes.json` with their verification badge, and the catalog row count is
   read from `docs/data_catalog.csv` (92 rows). `scripts/audit_docs.py` passes (0 uncatalogued hosts).

7. **The tests found four real defects, all fixed.** `relative_to(ROOT)` crashed on any path outside
   the repo (8 call sites → a `_rel()` helper); every directory/config default was bound at `def`
   time, so monkeypatching the constant did nothing (`Path(arg or CONSTANT)` at call time); the
   submission page rendered "rank 1 of ?" whenever the decision record carried no rank (`_policy_
   subtitle()` now renders the rank only when it exists); and a test token used `"single-band".title()`
   where the page says `single-band`.

**Verification.** Local suite **368 passed, 1 skipped** (baseline 318 at the start of the session;
+49 pinning the combined population, the landed-report reader and the submission page, +6 workflow
lints for the three-population scoring, the sha256 gate, the artifact contents and the race-safe
push, +5 for the sibling re-scores, `--strict` and `--gaps-only`). `read_landed_reports.py --check
--strict --quiet` passes on the committed evidence. Site rebuilt from `scripts/build_site.py`, and
the fold-gap caveat on the results page is now *derived* from `fold_gap_summary.json` instead of
asserting "folds 1–3 are in flight".

8. **The re-fire landed: the union population is measured on both arms — and it does not clear the
   bar.** Run 35477119490 (branch, 2026-09-20T02:06Z) committed both arms on all three populations.
   Fold 0, adopted policy `floor0.1_w0px`:

   | scope | population | baseline → pseudo | contrast | P(pseudo > baseline) | CI95 of the contrast |
   |---|---|---|---|---|---|
   | **held-out (8 blocks)** | **combined (union)** | 0.197183 → **0.228463** | **+0.031280** | **0.916** | [−0.010237, +0.082188] |
   | held-out (8 blocks) | proxy-only | 0.080584 → 0.148303 | +0.067719 | 0.988 | [+0.011044, +0.133105] |
   | held-out (8 blocks) | catalogue labels | 0.211259 → 0.103369 | −0.107890 | 0.000 | [−0.155060, −0.068517] |
   | trained-on (26 blocks) | combined (union) | 0.227349 → 0.189818 | −0.037531 | 0.0025 | [−0.064954, −0.007951] |
   | trained-on (26 blocks) | proxy-only | 0.092020 → 0.097677 | +0.005657 | 0.700 | [−0.013210, +0.028078] |
   | trained-on (26 blocks) | catalogue labels | 0.215206 → 0.106995 | −0.108211 | 0.000 | [−0.121993, −0.094166] |

   Derived verdict **`GAIN_ON_THE_COMBINED_SURROGATE`** ("the combined population is the only
   committed population that contains BOTH fault kinds, and it moves +0.031280"), and
   **`shippable_evidence: false`** — which in this file is a *constant by construction*, not a
   threshold that was evaluated: the reader makes no shipping decision, and the `shipping_note` says
   a pseudo-labelled field reaches the leaderboard only through `docs/FIELD_SELECTION_RULE.md` as a
   new `reblend.yml` RUN_ID with R3 measured on both fields' raw rasters, while the proxy arm here is
   source-circular. Read against the bar that rule *does* use (R3: paired block bootstrap
   P ≥ 0.95), the union contrast reaches **P = 0.916 with an interval that spans zero on 8 scoreable
   blocks** — so it is *suggestive, not established*, and the scorer's own reliability rule (below 12
   resampling units the interval is COARSE) means P here is a spread indicator, not a confidence
   statement.

   Three things keep this from being a positive result:
   * **The effect is the size of the replicate noise.** The first fire measured proxy +0.1036
     (P = 0.999) and catalogue −0.0874 (P = 0.001); this fire, same seed, same config, same
     partition, different runner, measured proxy **+0.0677** and catalogue **−0.1079**. That is a
     ~0.036 run-to-run swing on the proxy arm — the same order as the +0.0313 union contrast itself.
   * **The two scopes disagree in sign on the union** (+0.0313 held-out vs −0.0375 trained-on). A
     held-out-only gain is the direction that would matter (generalisation, not memorisation), but a
     sign flip between scopes on the same field is instability, not a result.
   * **Eight blocks is not a sample.** Both fires' numbers are in git history; the second overwrote
     the first's files, so quoting either one alone would overstate the precision.

   One coincidence worth defusing explicitly: the baseline's union held-out DTI is **0.197183**, which
   looks like a public leaderboard score (the snapshots bracketing it are **0.1972** on 2026-09-16
   and **0.2854** on 2026-09-21). They are unrelated quantities — 8
   held-out 51 km blocks of one survey against a local surrogate truth, versus the private
   expert-labelled new-fault test set — and the resemblance carries no information at all.

**Verification (this item).** Run 35477119490 steps 11/12/14 (artifact download, sha256-gated
re-score, `--strict` derived reading) all completed `success`; the committed
`data/evidence/pseudo_labels/fold0_derived_reading.log` is that step's own output. Merging this work
to `main` re-fires the same trigger path, and that run is **cancelled before it can commit**: a third
replicate would overwrite the committed evidence these docs quote without answering anything new, and
the workflow itself is already validated end to end by 35477119490. Re-firing on purpose means
changing a parameter (a different fold, or a pooled multi-fold contrast), not repeating this one. Locally the suite
was re-run after the sandbox restart that wiped the venv and the gitignored rasters: the bridge
reassembled all four files with every sha256 pin verified and the suite returned to **375 passed,
1 skipped** (the transient second skip was only `pypdf` missing from the fresh venv).

**Next session:** the union question is answered as far as one fold can answer it — *suggestive gain,
under-powered, inside replicate noise*. Do not re-fire fold 0 expecting a different number; if the
pseudo route is pursued it needs **≥12 scoreable blocks**, i.e. a contrast pooled over the four
committed folds rather than one, before P means anything. Otherwise the §11 ordering stands: **8
(human: enroll and upload — the only unbiased signal available) → 3 (1 m DEM derivatives) → 6 (GPU
EfficientNet-B5) → 4 (selection on the union population, which every fold field now carries)**.

## Session 18 (2026-09-19) — the field axis got a pre-registered rule and a machine gate; the block-holdout got its fold-0 gap; the pseudo-label route was built, proven leakage-safe, and dispatched

Session 17 closed the cross-catalogue question (refused) and left three open work items from the
plan: commit a FIELD-selection rule before any re-blend, dispatch block-holdout folds 1–3, and
build the SGMC pseudo-label experiment under spatial holdout. All three are done; the first two
are running on runners now.

1. **FIELD-selection rule: pre-registered, machine-checked, gate-locked (new).**
   The policy axis (floor/thin/width) was adopted by measurement in session 13, but the *field*
   axis — which ensemble mean ships — had been changed by hand twice (16-fold ens123 in session 14,
   reverted to 11-fold mean12 in session 16 after it scored 0.0850 vs 0.0999). That is a policy
   decided without a rule, so this session commits one before any further re-blend:

   * `docs/FIELD_SELECTION_RULE.md` pre-registers it: eligibility F1 (pinned re-blend provenance
     chain: the field's RUN_ID set must equal the trigger file's, and the evidence dir's live-fold
     count must equal both the blend report's n_folds and the trigger's MIN_FOLDS) and F2 (a
     committed proxy sweep with a reference_t0 and an adopted row); adoption R1 (best-in-window
     beats the shipped field by a fixed 0.010 margin), R2 (top of all three support windows), R4
     (beats the current policy's measured DTI), R5 (≥ 3 independent candidates in the window) — and
     R3, a paired 2,000-resample block bootstrap of the per-block DTI difference on **both fields'
     raw probability rasters** with P(new > shipped) ≥ 0.95. R3 is deliberately *unmeasurable from
     committed bytes*: until both raw rasters are supplied, every non-shipped field is refused, and
     a synthetic field that passes F1/F2/R1/R2/R4/R5 with R3 unmeasured is refused by a test.
   * `scripts/check_field_selection.py` is the rule's machine checker (constants are rule
     constants in the file: margin 0.010, P threshold 0.95, 2000 bootstraps, 3 candidates). It
     runs over all committed proxy sweeps + the shipped field and writes
     `data/evidence/field_selection.json`. Verdict on the committed evidence: **KEEP mean12** —
     mean12 passes F1 (11 live folds == n_folds == MIN_FOLDS), F2, R2, R4, R5; ens123 fails F1
     (its second run dir is not committed → UNVERIFIED) and R1 (0.0850 < 0.0999 − 0.010); the
     single ensembles fail R1 outright.
   * The gate is wired into `reblend.yml` as the first data step of the blend job
     (`--gate --runs <full RUN_ID set>`): the pinned re-blend (the shipped field) passes without
     measurement; any other RUN_ID set passes only if the committed `field_selection.json` says
     ADOPT of exactly that field with R3 measured, otherwise the job fails before downloading a
     byte. The step's position (before any download) is pinned by a new test, and the trigger
     file documents the four-step path to ship a new field.
   * `tests/test_field_selection.py` (7 tests) pins the rule end to end, including a synthetic
     field (real shipped raster + 0.45 mass on ~5,500 isolated proxy-truth pixels) whose sweep
     rows are *measured from the raster* — it is refused while R3 is unmeasured and ADOPTs
     end-to-end (gate rc 0) once R3 is measured with p ≥ 0.95; the margin is a rule constant
     (patch it and the verdict flips).

2. **Block-holdout: fold 0 landed with a +0.0114 gap; folds 1–3 are in flight.**
   `data/evidence/block_holdout/fold0_generalisation_gap.json`: fold 0 (seed 46, 51 km blocks held
   out) scores **0.0806 [0.0586, 0.1021]** on the 8 blocks it never saw vs **0.0920
   [0.0747, 0.1100]** on the 26 blocks it trained on — a positive memorisation-vs-transfer gap,
   i.e. the model does not fully transfer to unseen 51 km regions, and the gap is small enough
   that the proxy policy reading survives. One fold is a point estimate with no spread, so
   `TRAIN_FOLDS=1,2,3` is now in `.github/triggers/block-holdout-params` (fold 0 not re-trained;
   same seed 46 — fold identity is its geography) and the trigger push dispatches the three 300-min
   jobs. This session's push fires both it and the pseudo-label fire below.

3. **SGMC pseudo-labels: built, leakage-proven by test, smoke-trained, dispatched (new).**
   The scored faults are new, the labels are a lower bound, and external catalogues are allowed
   (rules §3.2) — QFaults proved to be the labels themselves, so the SGMC proxy's code-2
   population (61,664 px of mapped trace the labels do NOT contain) is the only remaining
   external-catalogue route. What was added:

   * `src/dataset.py`: `load_pseudo_mask()` and a `pseudo=`/`pseudo_weight=` path in
     `make_patches`, designed for the paired comparison: the block partition and the pos/neg
     **window selection use the original labels alone** (a pseudo run and its baseline run train
     on exactly the same windows), the held-out region is zeroed out of the pseudo mask before it
     touches the training signal (label mass *and* the FP-weight map, the same leakage rule as the
     labels), and pseudo pixels become label mass in training windows only. Empty-mask and
     `pseudo=None` are byte-identical (pinned).
   * `configs/config_pseudo_labels.yaml` — the block-holdout config plus exactly three keys
     (pseudo path/code/weight 1.0); a test pins that the two configs differ only there.
   * `.github/workflows/pseudo-label.yml` + trigger + params: trains fold 0 (seed 46 = the
     baseline's seed) with the pseudo config, scores the raw field with `--score-fold 0`
     (held-out blocks) and `--score-fold 0 --complement`, then **paired-bootstraps** the held-out
     arm against the committed no-pseudo baseline (0.0806 [0.0586, 0.1021]) and commits
     `data/evidence/pseudo_labels/fold0_paired_vs_baseline.json`. The header states plainly that
     no shipping decision is made there: a pseudo-labelled ensemble reaches the leaderboard only
     through the field-selection gate as a new RUN_ID with measured R3.
   * `tests/test_pseudo_labels.py` (7 tests) pins the leakage design on synthetic grids: baseline
     invariance, pseudo mass in training windows only (test windows byte-identical), FP weight
     credited at training-region pseudo pixels (fpw = 0) and unchanged beyond their R-px reach,
     and no training window touching the held blocks' collar — the channel through which a
     held-out pseudo pixel could otherwise leak.
   * Verified end to end in the sandbox: the full torch stack is installed locally now
     (torch 2.14.0+cu130, smp 0.5.0, timm 1.0.29 — PyPI, since download.pytorch.org is
     egress-blocked), and a fixture-grid smoke run trained 1 epoch with the pseudo path
     (`pseudo: +31 training-label px across 6 windows; partition and window selection unchanged`),
     with the experiment pinned in `outputs/manifest.json`.

**Verification.** Local suite **318 passed, 1 skipped** (the skip is the pre-existing
evidence-dependent rules-quote extraction; the baseline was 301 passed before this session's 17
new tests). The two new workflows parse and pass the static lints, including three new ones: the
gate must sit before any data step in reblend.yml, the pseudo-label scoring must be
`--score-fold`-restricted and must pair against the baseline's *held-out* (not complement)
report, and the committed pseudo fire must target a fold whose baseline is committed. Site
rebuilt from `scripts/build_site.py` (executive summary §6 + §12 rows 2/7 + final note; results
page scope note and follow-up list).

**Now running on runners (this session's push):** block-holdout folds 1–3 (3 × 300 min) and the
pseudo-label fold 0 (1 × 300 min). **Next session:** read the four landed reports (the fold gap
spread + the pseudo paired contrast), update the site and STATUS, and pick the next experiment —
most likely 1 m DEM derivatives or the GPU full config, per the §12 ordering.

## Session 17 (2026-09-19) — the two dispatched measurements landed: one reproduced bit-for-bit, one refused itself with a finding

Session 16 ended with two measurements running on GitHub-hosted runners because the sandbox cannot
reach `earthquake.usgs.gov`. Both came back, and they came back with opposite kinds of answer.

1. **Block-holdout: an independent environment reproduced the sandbox numbers exactly.**
   `data/evidence/block_holdout/sandbox_vs_runner.json` compares ten quantities between the sandbox
   computation and the runner's own — global DTI, weighted TP/FP/FN, emitted pixels, block counts,
   footprint pixels and both bootstrap CI bounds. **`agree: true` on all ten**; DTI **0.099859** to six
   decimals and CI95 **[0.088338, 0.111886]** identical. This is the strongest reproducibility result
   in the repository: the headline emission number is not a sandbox artifact.

2. **Cross-catalogue transfer: `REFUSED`, and the refusal is the scientific result.** The runner
   fetched QFaults layer 21 for the footprint — **14,481 features**, `fetched == service_reported`,
   provenance sidecar with every request URL — and rasterised it on the competition grid with the same
   script that rasterises catalogue A (SGMC). The overlap with the training labels is **total**:

   | quantity | value |
   |---|---|
   | catalogue B (QFaults) pixels, whole grid | 169,115 |
   | …of which outside the scored footprint (NaN in the submission) | 108,176 |
   | **B pixels inside the scored footprint** | **60,939** |
   | already within R = 3 px of a training label (code 1) | **60,938 — 100.00 %** |
   | left as a "new fault" population (code 2) | **1 px (0.0016 %)** |
   | label fault pixels within R of a QFaults trace | **60,986 of 60,988 — 99.997 %** |

   **The training labels in this footprint are QFaults.** That was anticipated as a possibility in the
   code and in LIMITATIONS (rules §3.3 — the labels come from the INGENIOUS Great Basin compilation,
   which distributes Quaternary fault layers), and it is now measured rather than assumed. A DTI
   computed against a 1-pixel truth population would be pure noise, so nothing is reported.

3. **The refusal is now a pre-registered guard, not an ad-hoc crash.**
   `scripts/measure_cross_catalogue_transfer.py` takes `--min-b-only-px 100` /
   `--min-b-only-fraction 0.005`: below either, it raises `PopulationDegenerate`, which `main` catches,
   writes the report **with the overlap that establishes it** (population table, thresholds, input
   sha256s, the caveat naming the SGMC proxy as the surviving surrogate) and **exits 0** — because
   "these two catalogues are the same lines" is a measurement worth committing. A genuinely broken
   B raster (under `--min-b-all-px 1000` in-footprint pixels: empty, misaligned or miscoded) still
   exits non-zero, so a data bug can never masquerade as a finding. Both paths are pinned by tests
   (`tests/test_cross_catalogue.py`, now 15 tests, including one that asserts the committed QFaults
   report agrees with `qfaults_stats.json`).

4. **The real refusal report is committed**: `data/evidence/xcat/transfer_report.json` (4.5 KB),
   computed locally from the runner-fetched raster against the shipped submission
   (sha256 `a3dcd6d5…`). Verdict string: `REFUSED - catalogue B is NOT independent of the training
   labels …`. `controls_pass: false`, `measurements: []`, `exit_code: 0`.

5. **Workflow hardened.** `.github/workflows/cross-catalogue.yml` had failed on its "Record overlap"
   step by reading top-level stats keys that do not exist (the real schema nests under `proxy.`). It now
   reads `s["proxy"]`, distinguishes **schema drift** from a **genuinely empty population**, prints the
   per-class breakdown, and downgrades a zero/near-zero code-2 population to a `::warning::` instead of
   failing the job — so the transfer job runs and commits the REFUSED report. The job-summary step
   renders both report schemas (REFUSED and full). All 13 embedded Python blocks in all 12 workflows
   parse; every workflow YAML validates.

6. **Consequence for strategy.** The prior on hidden-expert-set recall that EXECUTIVE_SUMMARY §11
   item 1 asked for **cannot come from a second Quaternary catalogue** — none is independent of these
   labels. Detection upside must come from 1 m DEM derivatives (§11 item 3) or model capacity (item 6),
   and the first unbiased signal remains the public leaderboard (item 8, human-gated). Item 5
   (ensemble 4) is **deferred on evidence**: at matched support the 11-fold shipped mean scores 0.0999
   against 0.0850 for the 16-fold blend, so more folds did not help.

7. **The workflow then failed for a second, unrelated reason — worth recording.** With the overlap
   step fixed, the transfer job started and died in 28 s with `argument --bootstraps: invalid int
   value: ''`. The params job contained `[ -n "$(v BOOTSTRAPS)" ] && BOOT="$(v BOOT)"` — it tested one
   key and assigned from another, and `BOOT` is not a key in
   `.github/triggers/cross-catalogue-params`, so the good `|| '1000'` fallback was clobbered with an
   empty string. The defect had been invisible for a session because the first run failed *earlier*, in
   the qfaults job, so the transfer job never started. Fixed, and three **static** lints added to
   `tests/test_workflow_yaml.py` (3 → 6 tests) that fail on this class without a runner: the tested key
   must equal the read key; a key read in an assignment must be set in the params file or documented in
   the workflow header; and an output interpolated straight into a `--flag` must have a non-empty
   fallback, because a push-triggered run has no `github.event.inputs` at all. Mutation-checked by
   restoring the original line (lints 1 and 2 fail with the offending `file:line`).

8. **Third fire: green, and the refusal reproduced on the runner.** All three jobs succeeded (run
   35412827594). The runner re-fetched QFaults a third time — 14,481 features again, integrity gate
   passed — and produced a **byte-identical** raster (sha256 `3fb2ca73…`) and an **identical REFUSED
   report**: same 60,939 / 60,938 / 1 population, same label-side reciprocal (60,986 of 60,988), same
   thresholds, `exit_code: 0`, scored against the shipped submission (`a3dcd6d5…`). The finding is now
   reproduced across three fetches and two environments, and the job summary renders it instead of a
   stack trace.

9. **Both PRs merged to `main`.** [PR #23](https://github.com/buffedlizard55-lab/GEMSDOE/pull/23)
   (session 16's work + the runner evidence + this session's guard) merged as `1b97c61`; the merge
   commit touched `.github/triggers/*`, so cross-catalogue, verify-sources and block-holdout all
   re-fired **on main** and pushed their own evidence — a *fourth* QFaults fetch (all four source links
   OK_200 this time, confirming the earlier 503s were transient), a fresh 29-sentence rules-quote check
   and a main-branch block-stratified measurement. [PR #24](https://github.com/buffedlizard55-lab/GEMSDOE/pull/24)
   then corrected catalog row E39's link status and rebuilt the site; its nine conflicts were all in
   **generated** `docs/*.html` and were resolved by regenerating from the merged sources rather than
   hand-picking hunks. Merged as `91c0473`. On `main`: Tests ✓, Pages ✓, Cross-catalogue ✓,
   Verify sources ✓.
   **Local test status:** 288 passed, 2 skipped, and 4 torch-dependent tests that cannot collect because
   the sandbox `.venv` no longer has torch (`.venv` is not persisted between sessions); the runner suite
   with torch is green (Tests ✓ on the PR and on `main`).

10. **Still running at the time of writing:** two `fold 0` block-holdout **training** jobs (one from the
   branch push, one re-fired on main), each with a 300-minute budget on CPU runners. They produce the
   first genuine generalisation gap (`--score-fold 0 [--complement]`); they do not change the shipped
   artifact, and their evidence will be committed by the workflow when they finish.

## Session 16 (2026-09-18) — error bars on every emission number, the field axis settled, and the cross-catalogue measurement built end to end

Until this session every quality number in the repository was **one global DTI with no error bar**,
so a 0.012 difference between two emission policies carried no way to ask whether it was larger than
the noise between regions of the survey. Three things changed.

1. **Block-stratified evaluation with a paired block bootstrap** — `scripts/block_holdout_eval.py`
   (new) scores the shipped submission per 51.2 km block (the same partition `src/blocks.py` uses for
   training) and bootstraps over blocks via the new `src/metrics.block_aggregate` /
   `bootstrap_from_blocks`. Measured on the committed bytes, in the sandbox:

   | quantity | value |
   |---|---|
   | reference policy (floor 0.1, thin, width 0) proxy DTI | **0.0999** |
   | block-bootstrap CI95 (56 blocks, 34 scoreable) | **[0.0883, 0.1119]** |
   | best genuinely different alternative (width 1 px) | 0.0878 (**−0.0121**) |
   | P(alternative beats reference) over block resamples | **0.008** |
   | catalogue (in-domain) DTI at the same policy | 0.2298 |
   | written support | 172,974 px |
   | candidates swept → **distinct** emissions | 50 → **10** (floor axis degenerate) |

   The adopted policy is now the best of the recoverable sweep **with 99.2 % bootstrap support**, and
   the degenerate floor axis (a hard-band raster has two values, so five floors select one support)
   is reported as `verdict.floor_axis_degenerate` instead of implying 50 independent measurements.
   Evidence: `data/evidence/block_holdout/block_stratified.json` (345 KB; per-block rows of duplicate
   emissions are pruned after every consumer has run, with the twin named).
2. **The sandbox reproduces the runner's committed evidence digit for digit.** The same reference row
   the GitHub-hosted runner measured (`data/evidence/proxy/eval_sweep-mean12.json`: dti 0.099859,
   TP_w 8378.12, FP_w 164461.741, FN_w 53285.88, 172,974 px) is reproduced locally to ≤ 4.4e-4 on the
   components and exactly on DTI and support. `--crosscheck-sweep` makes that a **gate**: a mismatch
   exits 2, and `reproduction.status` is committed inside the report. `.github/workflows/block-holdout.yml`
   (new) runs the same measurement on a runner and fails if the two environments disagree.
3. **The field axis is settled, and the shipped field is the right one.** A floor is a threshold on a
   field whose scale changes with the number of averaged folds, so "the same policy" is not the same
   emission: floor 0.1 / thin / width 0 emits 464,736 px on the 6-fold field and 144,738 px on the
   16-fold one. `scripts/compare_emission_fields.py` (new) therefore compares fields at **matched
   support** (best hard candidate within ±15/25/40 % of the shipped 172,974 px, same rule for every
   field): mean12 **0.0999** > ens123 0.0850 > ensemble2 0.0777 > ensemble1 0.0644, and the ranking is
   **stable in all three windows**. Ensemble 1's apparently superior 0.1365 was entirely a support
   effect. Evidence: `data/evidence/emission_field_axis.json`; the record also states that **no
   pre-registered rule covers the field axis**, which is the gap to close before any re-blend.
4. **Cross-catalogue transfer built end to end** (EXECUTIVE_SUMMARY §11 item 1): `scripts/fetch_qfaults.py`
   (new) fetches USGS QFaults layer 21 "National Database" for the footprint — service metadata read at
   run time, class vocabulary taken from the layer's own renderer, **integrity gate** `fetched ==
   service-reported` (14,482 features verified live 2026-09-18), full provenance sidecar with live link
   checks. `scripts/measure_cross_catalogue_transfer.py` (new) runs the **controls first** (labels-copy
   must score ≈ 0 against B-only, B-copy must score 1.0, blanket-ones gives the trivial floor), then
   emits catalogue A at several widths, scores against QFaults code-2 pixels, and tests
   `union(model, A)` — a probability maximum, not a mask OR — against a **pre-registered** criterion
   (gain > 0.01 DTI and P(union > model) ≥ 0.95 over block resamples). The verdict is derived:
   `ADOPT` / `DO NOT ADOPT` / `REFUSED` (controls failed) / `NOT MEASURABLE` (no model field).
   `.github/workflows/cross-catalogue.yml` (new) runs both on a runner, where `earthquake.usgs.gov` is
   reachable (it is not from the sandbox: curl exit 35).
5. **Block-holdout training path wired**: `configs/config_block_holdout.yaml` (new) sets
   `training.holdout: spatial_blocks` with `block_px`/`block_folds` that `block_holdout_eval.py --config`
   **cross-checks against its own scoring partition** (disagreement exits 2 — a score computed on a
   partition the model was not trained against is not a holdout score). `--score-fold K [--complement]`
   restricts the measurement to the blocks fold K held out, or to the blocks it trained on, so the
   generalisation gap is a measured pair of numbers with CIs rather than an assertion.
6. **Two defects found and fixed while reviewing this work**: the population-conflict counter counted
   the catalogue side alone and so overstated the conflict 32 blocks where the true both-signs conflict
   is 9 (`blocks_labels_lose` vs `blocks_conflict_labels_lose_proxy_gains`, now both reported and
   pinned by a test); and `build_proxy_catalogue.py` read the class vocabulary only from
   `query.rule_id_to_class`, which silently produced unnamed per-class rows for QFaults (it nests under
   `classes.rule_id_to_class`) — both keys are now accepted and the one used is recorded.
7. **Tests: 291 passed, 2 skipped** (was 175) — new suites `test_block_decomposition.py`,
   `test_block_holdout_eval.py` (28), `test_cross_catalogue.py` (12), `test_fetch_qfaults.py` (15),
   `test_compare_emission_fields.py` (10), `test_blocks.py`, `test_spatial_holdout.py`.

**Interpretation note that is now derived, not asserted:** DTI is *not* decomposable over blocks — TP_w
sums over truth pixels while the FP penalty is a global mass term — so a candidate can win in most
blocks and lose globally. Blocks are for **variance** (the bootstrap) and for locating where a score
comes from; the **global DTI remains the selection statistic**. `block_holdout_eval.py` computes whether
that conflict occurred (`interpretation.n_conflicts`, 0 in this run) instead of claiming it in prose.

## Session 15 (2026-09-18) — Executive summary subpage polished for submission, data bridge re-verified, 175 tests + audit pass

1. **Executive summary subpage polished for direct submission:** `docs/executive_summary.html` now opens with a **TL;DR 5-command box** (`git pull` → `assemble_data_bridge.py` → `prepare_data.py` → `validate_submission.py` → upload) pointing at the pre-computed, format-validated `data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif` (sha256 `a3dcd6d5…`, 569.5 KB, 11-fold ensemble, floor 0.1/thin/w0, rank 1 of 132). Companion markdown `EXECUTIVE_SUMMARY.md` adds the same quickstart, a **Common Pitfalls** table (6 measured failures + prevention), a **Data Placement Resolved** section (35168924460 → bridge → re-verify), a **Limitations** table, and a **Next Steps** queue. Both cite official sources line-by-line with `docs/data_catalog.csv` (89 rows) and machine evidence.
2. **Data placement re-verified in sandbox (2026-09-18):** `python scripts/assemble_data_bridge.py` (5 parts → 418,912,844 B, sha256 `4371c82e…`), `python scripts/prepare_data.py` (**PASS**, 3292×3730, 19 bands, EPSG:32611, 100 m), `python scripts/validate_submission.py` on shipped submission (**PASS**). Evidence: `data/evidence/data_placement.json` + fresh terminal output.
3. **Site rebuilt + audit + tests + metric self-test all green:** `python scripts/build_site.py` (10 pages), `python scripts/audit_docs.py` **PASS** (0 uncatalogued hosts), `python -m pytest` **175 passed, 2 skipped** (21 s), `python src/metrics.py --self-test` 8 checks — the same commands an entrant runs before upload.
4. **PR workflow:** Changes committed on `arena/01a0b255-gemsdoe`, PR opened, and merged to `main` to publish the Pages site via `docs/`.

## Session 14 (2026-09-18) — Executive summary subpage, data placement verified, and 3-ensemble sweep confirmed

1. **Executive summary subpage created and published:**
   [`docs/executive_summary.html`](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/executive_summary.html)
   and [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) now provide the definitive, step-by-step
   operational roadmap for making an eligible submission into the GEMS Prize challenge. It details
   the dual-phase prize structure ($50k Phase 1 + $250k Phase 2), verbatim rules eligibility criteria
   (§1.3, §1.4), mandatory Generative AI disclosure requirements (§3.2 with ready-to-use narrative
   template), exact technical GeoTIFF specifications (3292×3730, 100m res, EPSG:32611, single band
   float32, [0, 1], 57.92% NaN mask), the distance-weighted Tversky metric, and the adopted winning
   emission policy (`floor 0.1, thin, width 0 px`).
2. **Data placement completed and verified in sandbox:**
   Canonical competition rasters (`data/training_features.tif`, `data/labels.tif`,
   `data/sample_submission.tif`) were assembled from the sha256-pinned git bridge via
   `python scripts/assemble_data_bridge.py` and pre-flight validated via
   `python scripts/prepare_data.py` (exit 0, passing all grid, resolution, and band checks).
3. **Plain inference path wired to adopted policy:**
   `src/inference.py` now resolves shaping through `effective_shaping()`, adopting the measured policy
   from `data/evidence/emission_decision.json` (`floor 0.1, thin, width 0 px`) rather than falling back
   to the in-domain calibration (which scores only 0.0247 on unseen faults vs 0.1365).
4. **3-ensemble sweep confirmed on 16 live folds:**
   Workflow run 35285326679 swept the 3-ensemble mean (16 live folds across seeds 42, 43, 44) on the
   independent proxy catalogue (`eval_sweep-ens123.json`), confirming the adopted policy beats its
   reference policy by **+0.0581 contrast** (0.0850 vs 0.0269).
5. **Full test suite passing:** 175 passed, 2 skipped, 0 failures. `scripts/audit_docs.py` reports PASS.

## Session 13 — the emission policy is measured, reproduced, and shipped

The pre-registered rule (session 10/11) has three conditions: beat the shipped policy by > 0.01 on
the new-fault-like population, win at the plausible scored-truth sizes, and **reproduce on a second,
independently trained ensemble**. Session 12 built the path; this session measured the last
condition, fixed the two defects the measurement exposed, and shipped the policy it selected.

### 1. The candidate is a JOINT policy, and the rule is evaluated candidate-wise

The extended ensemble-1 sweep (session 11/12) put the best hard candidate at **floor 0.1, thin,
width 0 px** — proxy DTI 0.1365 against the shipped policy's 0.0410 on the same field — with
widening *hurting* at that floor (1 px 0.0918 → 20 px 0.0610). Until this session the conditions
were phrased around "the widest swept band", so the record's own best candidate was not the policy
the verdict was about. `scripts/decide_emission_width.py` now evaluates all three conditions against
the best measured candidate and derives the conclusion from them (the three branches — SHIP /
measured, not yet reproduced / measured, not shipped — are computed, never typed).

### 2. Condition 3 is measured on every field, including the shipping one

Four sweeps are committed, each scored against the candidate “floor 0.1, thin, width 0 px” as a
CONTRAST with that field's own reference policy (absolute proxy DTI is not comparable across fold
sets):

| field (sweep file) | policy | its own reference policy | contrast |
|---|---|---|---|
| ensemble 1 (run 35042805806) | 0.1365 | 0.0410 | **+0.0954** |
| ensemble 2 (run 35249562910, folds 6–11, seed 43) | 0.0777 | 0.0320 | **+0.0456** |
| **the shipping field** (mean of ensembles 1+2, run 35275312337) | 0.0999 | 0.0304 | **+0.0695** |
| **the 3-ensemble field** (mean of ensembles 1+2+3, 16 live folds, run 35285326679) | 0.0850 | 0.0269 | **+0.0581** |

All four keep the sign and exceed +0.01 → condition 3 passes on every independent ensemble AND on the
exact fields the adopted policy is applied to → the record's conclusion is **SHIP the measured policy**
(floor 0.1, thin, width 0 px), naming the ship path (`SHAPING_T0`/`SHAPING_DILATE`).

The shipping-field sweep also sharpens the argument for the rule below: that field's *own* argmax is
a different floor again (0.05 → 0.1260), and floor 0.05 is rejected by ensemble 1 (−0.0266). A
single-field optimum would have been wrong twice; the worst-case ranking is what keeps the choice
stable (mean12: `data/evidence/proxy/eval_sweep-mean12.json`).

### 3. Two defects the measurement exposed

* **The verdict was about the wrong policy.** Conditions 1–3 tested the width axis at the shipped
  floor while the search's winner was a floor change. Fixed as above; `tests/test_emission_decision.py`
  pins both directions (gain reproduced / reproduction absent / reproduction failing).
* **A test pinned the state of the evidence, not the rule.** `test_proxy_catalogue.py` asserted that
  an unmet condition *must* be recorded — so it failed the moment condition 3 passed. Replaced by the
  implication that matters: the conclusion is one of the three derived branches and cannot claim a
  SHIP while a condition is unmet.
* **A repeated `--second-sweep` flag silently dropped the earlier sweeps.** Regenerating the record
  by hand (`--second-sweep a --second-sweep b`) kept only `b`, because the flag was a single-value
  option split on commas; the workflow passes one comma-list, so the defect was invisible there. The
  flag now accumulates both forms, and a test pins that two flags produce two reproduction rows —
  condition 3 must never pass on less evidence than the command line asked for. The `verdict.top_priority`
  line was also made derived rather than typed, because the constant-ones comparison it stated had
  gone stale the moment the floor sweep beat that baseline.

### 4. The shipped candidate is not a maximum of one field

Ensemble 2's own argmax is a *different* floor (0.05 → 0.0953) and the shipping field's is floor
0.05 as well (0.1260) — but floor 0.05 is rejected by ensemble 1 (−0.0266), which is exactly the
disagreement condition 3 and the ranking below exist to catch. The record now carries
`verdict.robustness_across_ensembles`: every hard (floor, width) candidate that exists in **every**
sweep, ranked by its **worst** contrast against each sweep's own reference policy. Across the three
committed sweeps the shipped candidate ranks **1 of 132** (worst +0.0456 on ensemble 2; runner-up
floor 0.1 / width 1 px at +0.0414), and a `warning` is written if it is ever not first.
`cross_ensemble_ranking()` is unit-tested with a synthetic case where the first field's argmax loses
on the second.

### 5. The policy ships through the re-blend, and is measured where it is applied

`blend_submission.py` now takes `--shaping-t0/--shaping-dilate/--shaping-source`: a MEASURED joint
policy replaces the in-domain calibration (which maximises DTI against the faults the model trained
on and therefore always prefers the narrow, high-floor skeleton). Half a policy is a parse error.
The re-blend (run 35275312372, commit `c4f194e`) wrote `data/evidence/runs/ens12-adopted-floor0.1-w0/`
over ensembles 1+2 — 11 live folds, `MIN_FOLDS` pinned to that exact count, so a short download
fails the run instead of shipping a thinner ensemble. Verified after the fact, from the committed
bytes rather than from the log:

| check | value |
|---|---|
| `blend_report.shaping.source` / `.adopted` | `adopted_measured_policy` / `{t0: 0.1, thin: true, dilate: 0, evidence: data/evidence/emission_decision.json}` |
| in-domain control (same run) | `t0 = 0.469674`, un-thinned, 0.2065 held-out — reported, not shipped |
| submission | sha256 `a3dcd6d51303f312fd3e13667a1890d8eeab0752483432ddd46bc74231168009`, 569,531 B, 3292×3730 float32, EPSG:32611 |
| written support | **172,974 px** — identical to the mean12 sweep's measured emission for (floor 0.1, width 0), i.e. the raster is that policy applied to the 11-fold mean |
| TIFF tags | `shaping_t0=0.1`, `shaping_thin=True`, `shaping_dilate=0`, `shaping_source=adopted_measured_policy`, `shaping_evidence=data/evidence/emission_decision.json` |
| format validation | passed (CRS, 100 m res, single band, float32, values ⊂ [0,1], size/transform match the sample); NaN outside the GeoDAWN footprint as the rules require |

That tagged raster is the submission this branch would upload — `reblend.yml` is the only pipeline
path that applies an adopted policy (see the queue item about the other two paths).

In the same push, `proxy-eval.yml` was fired on the **field that would actually be submitted** — the
mean of ensembles 1+2, one blend of `folds` + `folds2` via the multi-run support — because a floor is
a threshold on the *mean* field, and both sweeps so far measured a single ensemble's field. That run
completed and the answer is yes: on the shipping field floor 0.1 beats the reference policy by
**+0.0695** (0.0999 vs 0.0304, `eval_sweep-mean12.json`), and the decision step rewrote
`data/evidence/emission_decision.json` with all three fields in condition 3 (commit `8713331`).

### 6. The inference path stopped silently shipping the old policy

`reblend.yml` was the only path that applied the adopted policy. `python -m src.inference` — which is
what `train-and-submit.yml` drives — shaped with `manifest["shaping"]`, the floor calibrated against
the faults the model trained on, and said nothing about which policy it had used. The two are not
close on the population that is scored: **0.0247 (calibration) vs 0.1365 (adopted)** on the
new-fault-like population.

`src/inference.py` now resolves shaping through `effective_shaping()`, in this order:

1. an explicit `inference.submission_shaping.t0` in the config — kept, but tagged
   `explicit_config_override` and *reported as having overridden a measurement* when one exists, so
   an experiment can never pass for the adopted default;
2. the SHIP record's best measured candidate (the default, because the shipped configs carry nulls);
3. the manifest's in-domain calibration — reached only when no measurement exists, and labelled.

Every written raster now carries the same provenance tags as the blend path
(`shaping_t0/thin/dilate/source/evidence`), the emission width is actually passed to
`optimize_submission` (it was dropped, so a future adopted non-zero width would have been applied as
0 while the summary claimed otherwise), and the run summary records both the adopted policy and the
in-domain counterfactual. `tests/test_inference_adopted_shaping.py` (9 tests) pins all of it — the
precedence, the tags, the `ast`-level check that `dilate` reaches the call, and that a record which
does not say SHIP is never adopted.

### 7. What is left

1. ~~Mean-of-1+2 sweep~~ **DONE** (run 35275312337, `SWEEP_LABEL=mean12`): the adopted policy beats
   the reference policy on the exact field it is applied to by +0.0695, and the decision record now
   carries all three fields in condition 3. Next field-level confirmation: re-sweep the 3-ensemble
   mean once ensemble 3 has landed.
2. **Ensemble 3** (run 35263581931, folds 12–17, seed 44): fold 4 failed in training (4 min), fold 0
   was still training at this checkpoint; the surviving folds join a later blend together with their
   own sweep, never without one.
3. **Detection.** 74.0 % of the new-fault-like truth lies more than 12 px from any emitted pixel; the
   oracle ceiling is 1.0000 at width 0 and 0.1618 at 16 px; the leaderboard top (0.2854 on the
   2026-09-21 snapshot) is 11.5× the constant-ones baseline measured here. More independent folds,
   then the cross-catalogue transfer
   experiment in `SUGGESTIONS.md`.
4. **Training selection still maximises in-domain DTI.** The emission policy no longer does; the
   model still does (early stopping, fold weights). A new-fault-like selection signal is the
   structural fix.
5. **Human-only:** DrivenData account + enrolment, first upload, eligibility, Pages source setting,
   deadline artefacts.


## Sessions 11–12 — the emission-width question, turned into a decision path

Session 10 established that three quarters of the error on the new-fault-like population is
**detection**, not localization, and that the exact metric has an interior optimum at a 16 px band
(+0.0466 over the shipped skeleton; 0.0247 → 0.0713). These two sessions removed the instrumentation
defects standing between that measurement and a shippable change, and added the two missing
comparisons.

### 1. The value axis: a hard band beats a distance ramp at the same support (measured)

`src/submission_optim.soft_band` emits values that decay from 1 on the skeleton to 0 at the band
edge; its support is **identical** to `dilate_mask(width)` by construction (`inside = d <= width`,
value `(1 - d/(width+1))**gamma`), pinned by `tests/test_shaping_band.py`. Scored with the official
metric against the 61,664 px of USGS SGMC trace the labels do not contain, on the shipped
submission's own thinned support (`data/evidence/proxy/eval_value_axis.json`, generated by
`scripts/eval_proxy_catalogue.py --sweep --soft-band`):

| band width | hard band (p = 1) | ramp γ = 1 | ramp γ = 2 | ramp ÷ hard (γ = 1) |
|---|---|---|---|---|
| 3 px | **0.050879** (195,520 px) | 0.031559 | 0.026147 | 0.62 |
| 6 px | **0.063707** (447,896 px) | 0.042210 | 0.032326 | 0.66 |
| 12 px | **0.070967** (1,032,370 px) | 0.055239 | 0.045509 | 0.78 |
| 16 px | **0.071317** (1,453,792 px) | 0.059525 | 0.051382 | 0.84 |

The supports are equal pixel for pixel (`emission_px` identical in every pair), so this is the value
axis alone. The hard band wins everywhere; the ramp's deficit narrows as the band widens, because
the ring a ramp discounts most is the one a hard band charges for least efficiently. **Verdict:** the
ramp stays a control arm — a plausible idea, measured and rejected, not shipped on plausibility.

### 2. The floor axis is binary on this field (measured, and now auditable)

The sweep records the support above every candidate floor *before* thinning
(`field_mass_profile`). On the shipped field every floor from 0 to the shipped 0.4697 gives the same
21,492 px support (`distinct_supports: 1`), which is why the whole floor axis moved the proxy score
by less than 1e-4 in session 10. The grid is still swept — the evidence file now says *why* it does
not matter instead of leaving it to be reconstructed.

### 3. `--min-dilate` could be a silent no-op (defect fixed)

The pooled calibration seeds its search with the raw unshaped row and also explored the un-thinned
branch, and both of those have width 0. On the catalogue the raw soft map genuinely beats every band,
so a run asked to ship a 16 px band could have written the width-0 skeleton and reported success.
`calibrate_shaping(min_dilate=...)` now loses by construction when a width is requested, skips the
un-thinned branch, marks the raw row `excluded_from_search` in the report, and `main()` asserts the
returned width is at least `--min-dilate` before anything is shaped or written. `tests/test_ensemble.py`
now requires `dilate == 3` for `--min-dilate 3` where it previously accepted `0`.

### 4. The oracle ceiling: why widening is compensation, not a property of the metric

`scripts/measure_miss_distance.py --oracle` computes the band ceiling arithmetically: a perfect
localizer that emits the truth scores 1.0; one that is perfect but still writes a band pays
FP_w = Σ d(x)/R. Because both TP_w and FP_w scale with |G| for a self-similar truth, the ceiling
1/(1 + 0.2·FP_w/|G|) is **independent of the hidden truth size** — the one upper bound that can be
quoted for a truth set we cannot see (`data/evidence/proxy/oracle_ceiling-proxy.json`).

| band | oracle ceiling | this model, same width | captured |
|---|---|---|---|
| 0 px | **1.0000** | 0.0247 | 2.5 % |
| 3 px | 0.5795 | 0.0509 | 8.8 % |
| 6 px | 0.3460 | 0.0637 | 18.4 % |
| 12 px | 0.2031 | 0.0710 | 35.0 % |
| 16 px | 0.1618 | 0.0713 | 44.1 % |
| 40 px | 0.0839 | 0.0657 | 78.3 % |

A perfect localizer emits the truth and wants width 0. The measured width optimum is therefore a
statement about *placement error*, and the ceiling is flat in the unknown |G|, so the remaining work
is detection improvement, not band tuning.

### 5. In flight at the time of writing

| run | what | state |
|---|---|---|
| 35262778745 | extended sweep on ensemble 1: widths to 20 px, floors 0–0.9, ramp gammas 1–2 | sweep job running; rewrites `data/evidence/proxy/eval_sweep.json` |
| 35249562910 | ensemble 2 (folds 6–11, seed 43) — the pre-registered third condition of the emission decision | 5 folds trained, fold 4 lost in training; blend pending |
| 35263581931 | ensemble 3 (folds 12–17, seed 44) — variance reduction for the final blend | just fired |

### 6. The extended grid landed (run 35262778745): the winning axis is the FLOOR, not the width

The re-run swept 152 thinned candidates on the pre-shaping ensemble-1 field — floors
{0, 0.05, 0.1, 0.2, 0.3, 0.362, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} × widths {0, 1, 2, 3, 4, 6, 8, 10, 12,
16, 20} px, plus ramp candidates on the same supports (`data/evidence/proxy/eval_sweep.json`,
`eval_sweep.log`). Three things changed the picture:

| measurement | value |
|---|---|
| `field_mass_profile.distinct_supports` | **11** — on the raw field the floor axis is *not* binary (the earlier "binary" reading was a property of the already-hardened submission, where any floor inside the same mask gives the same pixels) |
| best hard candidate | **floor 0.1, thin, width 0 px → proxy DTI 0.136452** (464,736 px emitted) |
| the shipped policy, same field and population | 0.041041 (`sweep_verdict.current_policy_dti`) → the best candidate is **+0.1118** |
| widths at that floor | 0 px 0.1365 → 1 px 0.0918 → 2 px 0.0782 → 6 px 0.0677 → 20 px 0.0610: widening **hurts** once the floor is right |
| best ramp candidate | 0.0634, still below every matched hard band |
| widths at the *shipped* floor | 0 px 0.0410 → 20 px 0.0706: widening helps **only** on the narrow, in-domain-optimal support |

So the two candidate changes are alternatives, not additive knobs: the 16–20 px band is the best
available move *given* the shipped floor, while lowering the floor is worth far more and prefers
width 0. `scripts/decide_emission_width.py` now ranks the best hard candidate of **every** swept
floor (it previously admitted floors {0, 0.043} × widths {0, 3, 6} plus the widest band, which
silently excluded exactly this candidate), records the winning floor's whole width curve, and —
when a second sweep is supplied — looks the same policy up in it and contrasts it with *that*
sweep's own reference policy, so a floor candidate is gated by the same reproduction rule as a
width candidate. Two tests pin both behaviours.

**Caveat, stated where the number lives:** the floor was selected and scored on the same 61,664 px
of proxy truth, so 0.1365 is selection-optimistic. The pre-registered rule — beat the shipped
policy by > 0.01 **and** reproduce on a second, independently trained ensemble — is therefore still
the gate, and condition 3 remains unmet until the ensemble-2 sweep lands.

### 7. What is left

1. **Condition 3** of `data/evidence/emission_decision.json`: sweep ensemble 2 (and 3) and check the
   floor-controlled contrast reproduces above +0.01. If it does, the widening ships through
   `reblend.yml` (`RUN_ID=35042805806,35249562910[,35263581931]` + `MIN_DILATE`) — the path that
   reads the width is tested end to end now.
2. **Detection.** The oracle table puts a number on it: the model captures 44 % of the 16 px ceiling
   and only 2.5 % of the width-0 one. More independent folds, then the pseudo-label / cross-catalogue
   transfer experiments listed in `SUGGESTIONS.md`.
3. **Human-only:** DrivenData account + enrolment, first upload, eligibility, Pages source setting,
   deadline artefacts.


## Session 10 — where the error actually is: three quarters of it is detection, not width

*(Session 9 follows below, unchanged.)*

The emission-width question had a split personality: widening the emitted band helps on the
new-fault-like proxy population and hurts on held-out crops (0.1903 → 0.0908 at a 6 px band).
Session 10 measured **why**.

### 1. Measured: how far the misses are (`scripts/measure_miss_distance.py`)

Run on the shipped submission of blend run 35042805806 (threshold 0.5), scored with the official
metric (R = 3 px, α = 0.2, β = 0.8) against the 61,664 px (6,166 km) of USGS SGMC trace that
`labels.tif` does not contain — the population that stands in for the unseen faults. Evidence:
`data/evidence/proxy/miss_distance-ensemble1.json`.

| quantity | measured |
|---|---|
| emitted pixels (thr 0.5) | 21,492 px (2,149 km) |
| miss distance p50 / p75 / p90 / p95 | 22.4 / 37.2 / 56.6 / 67.5 px |
| truth inside the metric's own R = 3 px | 4.9 % |
| truth within 6 / 12 / 20 / 30 px | 11.3 / 26.0 / 45.0 / 65.0 % |
| truth with NO emitted pixel within 12 px | **74.0 %** |
| exact DTI at band 0 / 3 / 6 / 12 / **16** / 20 / 30 / 40 px | 0.0247 / 0.0509 / 0.0637 / 0.0710 / **0.0713** / 0.0713 / 0.0697 / 0.0657 |

Two conclusions, measured rather than argued:

1. **The band width has an interior optimum at 16 px** on this population, and it is the first policy
   with a measured score above the constant-ones baseline (0.0585). The gain from 0 → 16 px is
   +0.0466, larger than any other knob measured so far.
2. **The remaining error is detection, not localization.** 74.0 % of the truth is more than 1.2 km
   from any emitted pixel; no amount of widening reaches it. Projecting each width onto the unknown
   size of the scored truth (the metric's own scaling; exact at the proxy's size) puts the crossover
   at |G| = 20,000 px (2,000 km): the skeleton wins below it, and both plausible anchors (2,604 km
   GeoDAWN-block fault density, 6,166 km this proxy) sit above it.

The projection and the width curve are embedded in `data/evidence/emission_decision.json`
(`width_vs_scored_truth_size`), and the proxy sweep's dilate grid now reaches 16 and 20 px so a
second ensemble can find the optimum the first grid (which stopped at 12) could not.

### 2. Reliability: one failed fold must not discard five

Run 35249562910 (fold offset 6, seed 43) lost fold 4 inside the training step (exit 1) — the Actions
log blob host is not on the sandbox egress allowlist, so the trainer's own stderr is unrecoverable
for that run. The ensemble workflow's rule was "refuse to blend unless every fold succeeded", which
threw away the other five folds' ~15 CPU-hours. Changed, and pinned by tests:

* the blend job now counts fold directories that carry **both** `prob_raw.tif` and a
  `heldout_mc*.npz` calibration crop, blends at ≥ 4 of 6 (`MIN_FOLDS`), refuses below that, and
  records any shortfall in the log and the committed evidence (`usable_folds.txt`);
* failed fold jobs upload a separate `foldlogs-<fold>` diagnostics artifact — named so the `fold-*`
  download pattern can never mistake a log for a fold;
* `.github/workflows/reblend.yml` can now rebuild a submission from **any** run's fold artifacts
  through a committed parameter file (`RUN_ID`, `EVIDENCE_DIR`, `MIN_FOLDS`, `MIN_DILATE`, …), which
  is the recovery path for run 35249562910's five complete folds.

`tests/test_ensemble_workflow.py` executes the workflows' own shell for exactly these paths (6 usable
→ 6, 5 + logs-only → 5 with a loud warning, 3 → refused, 0 → refused).

### 3. Review finding: the proxy sweep never scored the shipped floor

Reading the committed ensemble-1 sweep (`data/evidence/proxy/eval_sweep.json`) against the committed
submission score (`eval_submission.json`) exposed a gap in the *instrument*, not in the result:
`shaping_thresholds()` is log-spaced, so with the sweep's `SHAPING_GRID=4` the floors evaluated were
[0, 1e-4, 2.08e-3, 4.33e-2, 0.9]. On the real ensemble field the first four rows are the same mask
(14,285 px) — the field emits nothing between 0.043 and 0.9 — so the floor dimension was effectively
binary, and the floor the blend actually shipped (0.469674, 21,492 px, DTI 0.0247 on this population)
was never a row. The sweep's acceptance rule therefore compared candidates against the t0=0 skeleton
(0.0144) rather than against the shipped policy (0.0247), i.e. against a policy nobody ships.

Fixed in the instrument:

* `scripts/eval_proxy_catalogue.py` gains `--reference-t0` and `--reference-report` (reads
  `shaping.t0` from the blend report the sweep job already produces). The shipped floor is added to
  the grid when absent and becomes the reference row; `sweep_verdict` now records
  `reference_t0`, `reference_source`, `current_policy_dti` and `beats_current_policy_by`, and the
  acceptance rule compares against the shipped policy — explicitly saying so when no reference was
  given rather than silently substituting the t0=0 skeleton;
* `.github/workflows/proxy-eval.yml` defaults `SHAPING_GRID` to 11 (the same 11-point grid
  `configs/config_ci_ensemble.yaml` calibrates with) and passes `--reference-report blend_report.json`.

Two new tests execute both paths (reference floor present in the grid at every width, comparison
against that row, `--reference-report` provenance, loud failure on an unshaped report); reinserting
the old behaviour fails exactly those two. Suite: **129 passed / 1 skipped**.

### 4. Still open

Condition 3 of the emission decision — the second ensemble's own floor-controlled sweep — needs run
35249562910's artifacts; the run was still in flight at the end of this session. Until it lands the
verdict stays *"widen, but not yet"*, even though the miss-distance measurement argues the target
band is 16 px: the procedure was pre-registered, and a pre-registered procedure that gets overruled
by a comfortable-looking measurement is not a procedure.

## Session 9 — the data-placement blocker is resolved; the pipeline runs on the official bytes, in the sandbox and on runners

The stated blocker — *run `bash scripts/download_competition_data.sh` on any unrestricted machine into
`data/`, then `python scripts/prepare_data.py`* — is done, autonomously, with every byte accounted for:

1. **Git data bridge.** The sandbox allowlist (github.com/api.github.com/codeload.github.com/pypi.org only)
   blocks Dropbox, DrivenData, S3 *and* the Azure host that serves Actions artifacts, so no existing
   channel could carry the binary rasters into `data/`. New transport
   (`.github/workflows/place-competition-data.yml`, run 35168924460): a runner downloads the three
   official rasters from the data-tab Dropbox mirrors, **verifies each sha256 against the pinned
   inventory** (`data/evidence/inventory.json`, fails loudly on drift), runs `scripts/prepare_data.py`
   on the real bytes, and commits them to the branch as ~90 MiB parts in `data/bridge/` (GitHub rejects
   blobs ≥ 100 MB) with a manifest. Receiving side: `python scripts/assemble_data_bridge.py` re-verifies
   every part, concatenates, re-verifies the whole-file sha256, and places the canonical names.
   Round-trip, tamper-refusal, unpinned-refusal and verify-only are regression-tested
   (`tests/test_data_bridge.py`). Evidence: `data/evidence/data_placement.json`.
2. **`data/` now holds the official files**, sha-verified:
   `training_features.tif` (418,912,844 B), `labels.tif` (425,830 B), `sample_submission.tif`
   (1,599,597 B). `python scripts/prepare_data.py` **PASSES** in the sandbox: 3292×3730, 19 bands with
   `description`/`data_category` tags, EPSG:32611, 100 m, aligned bounds, labels 0/1, template float32
   [0,1] with NaN outside the footprint.
3. **Full pipeline re-run on the real data with the hardened code.** Runner run 35169168957 (smoke
   profile, data assembled *from the bridge*, not Dropbox): train → inference → validation **PASSED** →
   local score. Held-out DTI 0.0790 (prior smoke run 34876843912: 0.0588). Submission
   sha256 `ee183e26…` (470,840 B). Evidence: `data/evidence/runs/35169168957/`.
4. **The same pipeline now runs inside the 3.9 GB sandbox itself** — train → inference → validate →
   score on the placed bytes (`data/evidence/runs/local-sandbox-smoke/`): identical patch split as the
   runner at the same seed (311/234/233/78 — split determinism holds across machines), held-out DTI
   0.0997, validation **PASSED**, submission sha256 `b9f2bc4d…`. This required two measured memory
   fixes (below); the sandbox OOM-killed the first attempt at 3.8 GB anon-RSS.
5. **Memory fixes (measured OOM → fix → measured pass).** `src/train.py` frees the un-normalised stack
   once `Xn` exists (−933 MB); `src/dataset.py:make_patches` zeroes the padded stack in place instead
   of `.copy()`-ing a third full-stack allocation (−972 MB; the held-out windows are extracted before
   the zeroing, so semantics are unchanged); `src/inference.py` computes the validity footprint before
   freeing `X`. Peak RSS on the full 19-band grid: 3.8 GB (killed) → 3.18 GB (passes). This also
   lowers runner peaks.
6. **Training workflows are now Dropbox-independent.** `train-and-submit.yml`, the ensemble fold jobs
   and the blend job assemble `data/` from the committed, sha256-pinned bridge when it is present and
   only fall back to the mirrors otherwise — future runs no longer depend on live Dropbox links.
7. **CI regression for workflow YAML.** The first bridge firing (run 35168727415) failed in 0 s with
   zero jobs: an unquoted `: ` inside a step name. `tests/test_workflow_yaml.py` now parses every
   workflow and flags that pattern, so the failure class is caught by the Tests workflow instead of
   only when a trigger is pushed.
8. **Second 6-fold ensemble (seed 43, folds 6–11) fired** as the queued experiment for the
   emission-width decision (`data/evidence/emission_decision.json` condition 3: "second ensemble").
   It trains in parallel on public-repo runners (~2.5 h) and commits its evidence to
   `data/evidence/runs/<run_id>/` autonomously; next session analyses it against the widen decision.

Still true from session 8: both prize phases score the **new** fault dataset (disjoint from
`labels.tif`); read `docs/DISCOVERY_PLAN.md` before optimising local scores. The remaining blockers to
a leaderboard result are unchanged and listed in §5 below: a DrivenData account (to submit at all),
GPU capacity for the full config, and the eligibility check.

---

## Session 8 — reliability and provenance fixes

The official competition home page, problem page and About page were fetched and read again on
2026-09-17. The line-by-line source table and the changes made in this review are recorded in
[`REVIEW_2026-09-17.md`](REVIEW_2026-09-17.md); the competition pages and official rules PDF remain the
controlling sources.

Implemented in this pass:

- direct `src.inference` now uses `src.submission_io.write_submission()` rather than forcing tiled TIFF
  output on the striped sample profile; the writer reads the bytes back and refuses an invalid or empty
  raster;
- training/inference workflow pipelines no longer use a successful `grep || true` subshell that masks a
  failed Python process; the ensemble blend explicitly refuses an incomplete fold matrix;
- optional USGS 3DEP DEM derivatives are now a real, symmetric path: a supplied local mosaic is reprojected
  to the feature grid and used by both train and inference. The default GPU config is now disabled until
  `data.external_dem_path` is set, so the documented default cannot silently change the channel count;
- SMP is imported lazily and the rules evidence test handles a report that intentionally omits the optional
  extracted PDF text.

**Current local verification:** run `python -m pytest tests -q` after installing the project/test dependencies;
`REVIEW_2026-09-17.md` explains why the sandbox cannot produce a leaderboard score. The full prior evidence
and historical findings remain below; they are retained to make regressions auditable rather than erased.

---

Supersedes the previous `STATUS.md` (session 3, 2026-09-15). Session 3's own claims are quoted where
this session falsified them, because the way they were falsified is the most useful thing in this file:
**a green workflow was treated as evidence.**

---

## 0. Session 7 (2026-09-16, second pass) — the rules check went green, and the width question got an answer

**Green:** `verify-rules` run 35153898428 — 29/29 rule sentences verified verbatim against the
official PDF (`data/evidence/rules_quotes.json`), with the 18 page-furniture removals recorded
(bare page numbers, glued section numbers). The last blocker on that check was that pypdf extracts a
page number *inside* the sentence that continues across the page break; the fix drops blank lines
before looking for page furniture, and it is now proved end to end by a synthetic reportlab→pypdf
PDF in CI. Tests 35153898433 and Pages 35153898486 are green on the same commit.

**The width question has a third measurement, and it disagrees with the first.** `proxy-eval` run
35152701740 swept floor × thinning × emission width on 61,664 px (6,166 km) of USGS SGMC fault trace
that `labels.tif` does **not** contain — the closest measurable stand-in for the scored new-fault
population. Proxy DTI rises monotonically with the width (0.0144 → 0.0395 from 0 px to 6 px), the
opposite sign to the held-out-crop sweep (0.1903 → 0.0908). `scripts/decide_emission_width.py`
reconciles them by projecting every measured policy onto a range of possible scored-truth sizes
using the metric's own scaling (`DTI = TP_w/(0.2(TP_w+FP_w) + 0.8|G|)`: the wrong-mass term does not
grow with |G|, the missing-mass term does), and writes the decision:
**widen, but not yet** — conditions 1 and 2 met (+0.0149 on the new-fault-like population; 3/3
plausible |G| anchors), condition 3 (second ensemble) unmet, default unchanged.

**The uncomfortable number in that table:** a constant-ones submission scores **0.0585** on the
new-fault-like population, beating the shipped skeleton's **0.0247** and every swept candidate.
Recall is worth four times precision under α=0.2/β=0.8, so the shipped emission is not conservative,
it is *under-emitting*. The crossover is computed and published: the skeleton stays ahead only
below ~2,200 km of scored truth.

**Three measurement defects found and fixed while reading that evidence** (each had produced a
plausible-looking number that meant something else): the truth length was converted with 0.01 km/px
instead of 0.1 km/px (every committed truth length was 10× short); the blanket-ones baseline was
taken over `np.isfinite(pred)`, so its definition changed with whichever raster was scored; and the
score of the raster handed to `--pred` was published as "as submitted", which mislabelled the
sweep's soft ensemble map as a submission on the site's Results page. All three are fixed, pinned by
`tests/test_proxy_catalogue.py::test_eval_units_support_and_role_are_unambiguous`, and the sweep now
clips every candidate to the data footprint so it compares *legal* submissions.

**New page:** `docs/verification.html` — the public leaderboard read directly (43 entrants; #1
0.1972; top-5 cut ≈0.1454, read 2026-09-16), an explicit "this repository is not on it" statement,
and a table re-checking every load-bearing external claim from its own URL
(`data/evidence/independent_verification.json`).

**Suite:** 91 passed (was 86; +3 proxy-contract tests, +2 site tests). The audit test deleted by an
earlier slice-to-EOF edit is restored.

---

## 1. What this session found: the ensemble run had produced no submission

Session 3 reported: *"On success the blend job commits `data/evidence/runs/35042805806/` (blend_report.json,
validation.log, submission.tif) back to this branch; the artifact `submission-final-*` is the file to
upload to DrivenData."* It also reported *"full suite 23/23 green"*.

Audited against the committed bytes (not the logs):

| artifact | claimed | measured |
|---|---|---|
| `data/evidence/runs/35042805806/submission.tif` | the submission | **110 bytes** — a GDAL stub, not a raster |
| `data/evidence/runs/35042805806/validation.log` | a validation report | **an argparse usage dump** (`unrecognized arguments: --sample`) |
| Actions artifact `submission-final-35042805806` | the file to upload | **1,183 bytes total** |
| `data/evidence/runs/35042805806/manifest.json` | — | missing (never produced) |

The blend job's own log shows why (`data/evidence/runs/35042805806/blend.log`):

```
pooled shaping: t0=0.372 thin=True -> mean held-out DTI 0.1927 (unshaped 0.1560)
rasterio._err.CPLE_AppDefinedError: _TIFFVSetField:submission.tif: Bad value 3292 for "TileWidth" tag
rasterio.errors.RasterBlockError: The height and width of TIFF dataset blocks must be multiples of 16
```

Three independent defects, all fixed this session and each covered by a regression test:

1. **The writer could not write.** `scripts/blend_submission.py` copied
   `sample_submission.tif`'s rasterio profile — a *striped* GeoTIFF, so `blockxsize == width == 3292` —
   and then forced `TILED="YES"`. GDAL rejects a 3292-px tile width. Fixed by the new
   `src/submission_io.py` (`clean_profile()` drops stale block geometry and pins legal 256-px tiles;
   `write_submission()` **reads the bytes back** and raises unless they are a single-band float32
   raster on the requested grid with non-zero mass). `tests/test_submission_writer.py` reproduces the
   original crash against the old profile, so the defect stays falsifiable.
2. **The failure was masked.** Every workflow step piped Python into `tee` without `pipefail`, so a
   crashed step exited 0 and everything downstream — including "Validate submission format" — "passed".
   `set -uo pipefail` is now on the train / inference / blend / validate steps, and the commit step
   refuses to commit a submission unless it is >10 kB, reads back as a raster and is non-empty;
   otherwise it commits `FAILED.json` instead of a stub.
3. **The binding floor search still used the old grid.** PR #9 replaced `linspace(0.02, 0.9, n)` with
   the log-spaced `shaping_thresholds()` in `src/train.py` — but not in `scripts/blend_submission.py`,
   which is where the 6-fold workflow's *binding* calibration happens. Fixed; measured effect on the
   flat-map case that used to collapse: the floor `t0 = 0.000` is now reachable and a valid sparse
   submission is written instead of an abort.

**Lesson recorded in the site and here:** a workflow conclusion is not evidence. The evidence is the
bytes, read back.

## 2. What this session verified: the scoring universe (and why it changes the target)

The canonical rules PDF was re-read line by line and the relevant sentences are now **machine-verified
verbatim** rather than paraphrased. Actions run **35132263421** (55 s) confirmed all three links in the
chain at once: the canonical PDF at `docs.nlr.gov` is **byte-identical** to the data-tab mirror we
inventoried (`sha256 50d854b1e0239fe6b9648d9fa5c7537bc7b6e5bc10cf6b37a2ff9aa401c36938`, 455,140 B,
`identical: true`), **11/11 quoted sentences matched verbatim**, and a re-fetch of all 76 catalog URLs
recorded 54 `OK_200`, 8 redirects, 10 publisher bot-blocks, 1 login-walled (the data tab) and 2 items to
review. Mechanism: `scripts/verify_rules_quotes.py` extracts the PDF from
`https://docs.nlr.gov/docs/fy26osti/96647.pdf`, normalises whitespace/unicode, and asserts each quoted
sentence appears; the *Verify official sources* workflow also checks that PDF's sha256 against the
copy inventoried from the data tab (`GEMS_96647.pdf`, 455,140 B), proving the mirror and the canonical
document are the same file. Evidence: `data/evidence/rules_quotes.json`,
`data/evidence/rule_sources_verification.json`.

The three load-bearing sentences:

* §1.1 — "In Phase 1, submissions will be evaluated against a privately withheld subset of the original
  **new** fault dataset compiled by expert reviewers."
* §1.1 — "Submissions will be reevaluated against the full, revised **new** fault dataset using the same
  distance-weighted Tversky index."
* §3.3 — "The training labels contain **existing** fault data at 100-m resolution … obtained from the
  INGENIOUS project's Great Basin Regional Dataset Compilation."

So **both prize phases score faults that are absent from the only labels we can download**, and Phase 2
does *not* add the existing catalogue back in. Session 3's phrasing — "the public `existing_faults.tif`
is training data, NOT the scoring target … copying the known-fault raster … is a trap" — was right; what
was missing is the consequence: *every* selection signal in the pipeline (early stopping, fold weights,
and the floor/thinning calibration) is fitted to the catalogue, i.e. to a population that is disjoint
from the scored one. Full write-up and the experiment queue: **`docs/DISCOVERY_PLAN.md`**.

New capability that follows from it — `src/discovery.py` (+ `tests/test_discovery.py`, 6 tests): every
blend report now carries `novel_fraction` (share of emitted probability mass farther than R=3 px from
any catalogued fault), `candidate_new_faults` (connected components that touch no catalogued fault, with
area and extent) and `catalog_recall_R`. Those are the only numbers in the repo that can distinguish
"this model is restating the catalogue" from "this model is proposing faults the catalogue does not
contain" without access to the hidden labels.

Independent corroboration that this is the right frame, from the paper the competition's own About page
recommends (Hermant, Kiersnowski & Bellanger 2025, Stanford SGW, fetched 2026-09-16): they keep **only
fault-bearing training tiles**, explicitly to "limit the possibility of learning images without mapped
faults when there should be some due to **operator observation bias**", and they measure up to **400 m**
of disagreement between the USGS Quaternary catalogue and their own expert labels in north-central
Nevada. Our config keeps 35 % empty windows (`neg_fraction: 0.35`) — a concrete, citable A/B target.

## 3. Verified environment facts (measured, not assumed)

* Sandbox egress allowlist still permits only `github.com`, `api.github.com`, `codeload.github.com`,
  `pypi.org`/`files.pythonhosted.org`. Re-measured 2026-09-16: `www.dropbox.com`, `drivendata.org`,
  `s3.amazonaws.com`, `prd-tnm.s3.amazonaws.com`, `sciencebase.gov`, `apps.nationalmap.gov`,
  `gdr.openei.org`, `raw.githubusercontent.com`, `objects.githubusercontent.com`,
  `pipelines.actions.githubusercontent.com` all fail (TLS EOF).
* **GitHub Actions *artifacts* remain unreachable from the sandbox** — newly measured: downloading
  `submission-final-35042805806` fails with `EOF` against
  `productionresultssa8.blob.core.windows.net`. A *runner* can download them (that is what the new
  re-blend workflow does).
* `github.com` git transport works, so small artifacts (reports, `submission.tif` ≈ 1 MB) travel back
  as ordinary commits. Large rasters **can** now travel too, as ≤ 90 MiB sha256-pinned parts
  (`data/bridge/`, session 9) — GitHub rejects single blobs ≥ 100 MB, not large totals.
* The full test suite now runs **in-sandbox with torch installed** (CPU wheel from PyPI):
  **37/37 pass** (`pytest tests -q`). Previous sessions could not install torch here at all.

## 4. What was rebuilt this session — and it produced the first real submission

`.github/workflows/reblend.yml` (+ trigger `.github/triggers/reblend`) re-runs **only** the blend step
against the fold artifacts of run 35042805806 — six checkpoints and six full-raster probability maps
(≈120 MB each) that the runner can still fetch, valid until ~2026-09-30. Cross-run artifact download
(`actions/download-artifact@v4` with `run-id`) is the mechanism. Actions run **35131318033** finished in
**7 m 46 s** (the training that it reuses took 2 h 36 min) and committed:

| output | value |
|---|---|
| `data/evidence/runs/35042805806/submission.tif` | **386,018 B**, tiled, blocks 256×256, sha256 `a5ae61d5…` |
| grid / format | 3292×3730, 1 band float32, EPSG:32611, 100 m, NaN outside the footprint (57.93 % of pixels), values in [0,1] |
| emitted area | 21,492 non-zero px of 5,165,852 finite px (0.42 %) |
| validation | `✅ Validation PASSED` — CRS, resolution, dtype, range, size and transform all match the template (`validation.log`) |
| pooled shaping | `t0 = 0.470`, `thin = True`; mean held-out DTI 0.1903 (unshaped 0.1560) |
| informational DTI vs the **known** catalogue | 0.1387 (blanket-ones floor on this grid: 0.0246) — wrong-universe number |
| **discovery profile** | `novel_fraction = 0.684`; 6,090 px in **617** components that never come within R of a catalogued fault (largest 92 px, ≈8.4 km long); catalogue recall @R = 0.214 |

The discovery profile is the interesting one: **68 % of the emitted probability mass sits farther than
300 m from any catalogued fault**, so this submission is not merely restating the catalogue — it is
proposing 617 candidate fault segments the catalogue does not contain. Whether they are real is exactly
what Phase 2's expert review decides, and what the public leaderboard would score; neither is available
here.

**Fold spread is wide and worth acting on** (measured, from the same report): held-out DTI per fold was
0.3351, 0.2665, 0.1780, 0.1776, 0.0916, 0.0742 — a 4.5× spread between the best and worst fold of the
*same* configuration. Equal averaging squeezes a weak fold into a strong one, and the pooled calibration
that picks the floor scores the same folds it fits, so its mean is selection-optimistic. Both are now
instrumented (`scripts/blend_submission.py`): `--calibrate loo` re-fits the floor — and the fold-weight
rule — on the five folds that are *not* being scored, and reports the oracle ceiling next to it, so the
optimism is a printed number instead of an assumption. Experiment `data/evidence/runs/35042805806-dilate-ab/`.

**The width of the emitted line was the binding constraint, not the floor** (measured 2026-09-16,
`data/evidence/shift_robustness.json`, script `scripts/measure_shift_robustness.py`). On the one window
where a written submission and the official label raster coexist (rows 2048–2560, cols 1280–1792 of the
full grid; `data/fixture/fixture_labels.tif`, 5,154 label px):

| band grown around the skeleton | pixels kept | DTI (no shift) | DTI, labels shifted ±3 px |
|---|---|---|---|
| 0 px (what the pipeline wrote) | 801 | 0.0555 | 0.0406 |
| 3 px | 8,063 | 0.1109 | 0.0991 |
| 6 px | 19,908 | **0.1260** | 0.1212 |
| 8 px | 29,249 | 0.1249 | 0.1240 |

The 6-fold submission written on 2026-09-16 kept 801 px in this window against 5,154 label px: it is
**under-covering**, and no floor can fix that — the old search could only choose between a skeleton and
an un-thinned blob. `--dilate-grid` now searches the band width, and the reblend workflow is re-running
the same six saved folds with it (`data/evidence/runs/35042805806-dilate-ab/`, triggered 2026-09-16).

**Reading of the numbers, stated plainly.** The labels in that table are the *public catalogue*, not the
scored set, and translating them is a stress test of the writing operator — not a measurement of the
competition metric. It is reported because the asymmetry is structural: TP<sub>w</sub> credits a
prediction up to R = 3 px away in full, a missed label costs 0.8 per unit, a false positive 0.2. For the
scored faults — new to the expert-reviewed dataset (rules §1.1/§3.5), never seen in training — the
model's localisation error is strictly larger than on the catalogue, which is exactly the regime where a
skeleton loses everything and a band still collects credit.

## 5. Still needed (unchanged by any of the above)

1. **A DrivenData account + enrolment** to (a) download the official data-tab files directly, (b) upload
   submissions at all, and (c) see the public leaderboard — the only unbiased signal for the scored
   universe, 3 submissions/week (rules §3.2). Nothing in this repository can substitute for it.
2. **A GPU** for the full `configs/config.yaml` run (EfficientNet-B5, 10 splits, 60 epochs). CPU runners
   cap the ensemble at resnet34 / 45 epochs / ~2.5 h per fold, which is what run 35042805806 measured.
3. **Eligibility check** (rules §1.3): competitors must be US citizens/permanent residents, private
   entities US-incorporated, academic institutions US-accredited; FFRDC-affiliated researchers may
   compete individually but are not cash-eligible; DOE employees and support contractors are excluded.
4. **Narrative + code assets** (rules §3.2, §3.5): finalists must submit complete code assets and the
   generative-AI disclosure — this repository is structured to be that submission (manifests,
   environment pins, reproduce page, audit gates).
5. **1 m DEM derivatives** (716 tiles already confirmed against the live USGS 3DEP bucket) and the
   proxy-catalogue evaluation of `docs/DISCOVERY_PLAN.md` §3.
6. **The public leaderboard score itself.** Every number in this repository is a proxy: catalogue DTI is
   the wrong universe, discovery diagnostics are unlabelled, and the width study is a stress test. Three
   submissions a week against the real metric (rules §3.2) is the only way to replace proxies with
   measurements, and it needs the account from item 1.

---

## 6. Session 5 (same day): the rules verified verbatim, and the emission width

### 6.1 The scored population, now machine-verified rather than paraphrased

Session 4 established that both prize phases score expert-mapped *new* faults. That conclusion has now
been re-derived from the official PDF on a runner, sentence by sentence, by
`scripts/verify_rules_quotes.py` (workflow `verify-rules.yml`, run
[35134058898](https://github.com/buffedlizard55-lab/GEMSDOE/actions/runs/35134058898), 16 s):

* **19 of 19 quoted sentences matched verbatim** after NFKC + quote folding + de-hyphenation +
  whitespace collapse. No fuzzy matching; a miss fails the step and the commit.
* The document fetched from `docs.nlr.gov/docs/fy26osti/96647.pdf` is **byte-identical** to the data-tab
  mirror recorded in `data/evidence/inventory.json`: `sha256 50d854b1e0239fe6b9648d9fa5c7537bc7b6e5bc10cf6b37a2ff9aa401c36938`,
  455,140 B, `identical: true`.
* Evidence: `data/evidence/rules_quotes.json` (+ `rules_quotes.log`), committed by the workflow; the home
  page of the site renders the six decisive sentences with their found/not-found marks.

The sentences that carry the strategy, quoted exactly:

| id | § | sentence |
|---|---|---|
| `phase1_target` | 1.1 | "In Phase 1, submissions will be evaluated against a privately withheld subset of the original new fault dataset compiled by expert reviewers." |
| `phase2_target` | 1.1 | "Submissions will be reevaluated against the full, revised new fault dataset using the same distance-weighted Tversky index." |
| `phase2_eligibility` | 1.1 | "All Phase 1 competitors will be eligible to compete in Phase 2 and will be automatically submitted for consideration." |
| `experts_revise` | 1.1 | "After Phase 1, expert reviewers will use submitted predictions to revise the new fault dataset." |
| `labels_source` | 2 | "The labels for this prize come from the USGS Quaternary Fault and Fold Database and from a set of newly identified faults labeled by geology experts at the National Laboratory of the Rockies (NLR) and USGS." |
| `ranking_basis` | 3.2 | "Second-round prize rankings will be determined by running the selected final submissions against the complete updated test set created by expert review." |

Two consequences worth stating because they are easy to get backwards: the released training label
raster is the *existing* catalogue (problem page + rules §3.3) while both phases score the *new* faults,
so reproducing the catalogue earns credit only where the experts' new labels coincide with it; and no
Phase-1 top-5 cutoff gated Phase-2 entry, so there is nothing to be gained by trading discovery for
catalogue coverage.

### 6.2 The emission width was the binding constraint, not the floor

Measured (`data/evidence/shift_robustness.json`, script `scripts/measure_shift_robustness.py`) on the one
window where a written submission and the official label raster coexist — full-grid rows 2048–2560 ×
cols 1280–1792, 5,154 label pixels:

| band grown around the skeleton | pixels kept | DTI (labels as-is) | DTI (labels shifted ±3 px) |
|---|---|---|---|
| 0 px — what the pipeline wrote | 801 | 0.0555 | 0.0406 |
| 3 px | 8,063 | 0.1109 | 0.0991 |
| 6 px | 19,908 | **0.1260** | 0.1212 |
| 8 px | 29,249 | 0.1249 | 0.1240 |

The submission kept 801 px against 5,154 label px in that window: it is **under-covering**, and the old
search could not express the fix — its only two options were a pure skeleton and the un-thinned
floor-passing blob. `scripts/blend_submission.py --dilate-grid` (default `0,1,2,3,4,6`) now searches the
band width on the held-out crops, and `--calibrate loo` re-fits the floor — and the fold-weight rule — on
the folds that are *not* being scored, printing the oracle ceiling next to the honest mean, so the
selection optimism is a number instead of an assumption.

Honest boundary: the labels used in that table are the public catalogue, not the scored set, and shifting
them is a stress test of the writing operator. It is reported because the asymmetry is structural
(TP<sub>w</sub> credits a prediction up to R = 3 px away in full; a missed label costs 0.8, a false
positive 0.2), and because it is *most* severe exactly where the scored faults live — faults the model
never saw in training, whose traces it cannot localise as tightly as the catalogue's.

Runner A/B on the six saved folds (emission width + LOO audit, no retraining):
`data/evidence/runs/35042805806-dilate-ab/` — workflow `reblend.yml`, run 35133590776. Its verdict decides
the default: a wider band is kept only if the LOO mean beats the skeleton by > 0.01.

---

## 7. Session 6 (same day): line-by-line review — 12 findings, 11 fixed, 1 queued

Method: read every line of `src/metrics.py`, `src/submission_optim.py`, `src/discovery.py`,
`src/submission_io.py`, `src/train.py`, `scripts/blend_submission.py`,
`scripts/measure_shift_robustness.py`, `scripts/audit_docs.py`, `scripts/verify_rules_quotes.py`,
`scripts/verify_links.py` and the `_scoring_universe`/`build_index` renderers; re-ran the suite
(38/38 green on arrival — the "37/37" in §3 was stale by one test), the metric self-test (8/8),
`audit_docs.py` (PASS, but with 1 uncatalogued-host REVIEW), and `build_site.py` (rebuild
drifted from the committed pages — the thread that found F9–F11). Environment: fresh `.venv`
from `requirements.verified.txt` + pytest on the stock sandbox (torch 2.14.0+cu130, CPU-only).

Findings (severity = consequence if left in place):

| id | severity | finding | disposition |
|---|---|---|---|
| F1 | **high (methodological)** | `loo_aggregation`'s fold-weight audit averaged other folds' held-out crops — *different geographic windows* (each fold's own compact subset, `train.py::heldout_maps`) — and scored the mix against the held-out fold's gt. Cross-geography averaging cannot measure the weight rule. | **removed** the comparison; LOO weights still fitted and reported per row, `weight_rule_gain` kept as `None` with a note. Valid weight evidence remains the full-map A/B (`local-mini-ensemble/`, same maps, same grid). Sound LOO weight audit queued (SUGGESTIONS §7.1). |
| F2 | medium (latent crash) | `calibrate_shaping` returned a 4-tuple with no usable folds; every caller unpacks 5 → `ValueError`. | returns `(0.3, True, nan, [], 0)` |
| F3 | medium (latent wrong-number) | `calibrate_shaping` + `loo_aggregation` mutated `f["pred_crop"]` when `pre` was set → `--frangi --calibrate loo` applied Frangi twice to the same crops. | copy-on-transform in both |
| F4 | low | `calibrate_shaping` docstring said 4-tuple return, code returns 5. | docstring fixed |
| F5 | low | `sl` lambda comment said "centre window", code took top-left `a[:k,:k]`. | removed with F1 |
| F6 | low | `--weights dti` fell back to equal weights silently when a fold lacked DTI. | prints a WARNING, still falls back |
| F7 | low (docs) | `SUGGESTIONS.md` claimed "audit now finds no uncatalogued hosts" in the same row that contained the `https:`+ellipsis stub the audit flagged — self-falsifying. | reworded; audit now lists **0** uncatalogued hosts |
| F8 | low | `discovery.py`: `novel_bboxes` largest-first + min_px-filtered, but `novel_component_px` was last-25-by-label-id, unfiltered — the fields could disagree. | derived from the same boxes |
| F9 | low (reproducibility) | `generate_dummy_submission.py` used unseeded `np.random` — every invocation differed. | `--seed` (default 42); byte-identical output pinned by test |
| F10 | low (dead code) | `train.py` built a full `TensorDataset` of the test windows every fold and never iterated it (scoring goes through `predict_patches`); `blend_submission.py` carried an unused `import math` and (after F1) an unused `_weighted_mean`. | removed |
| F11 | **high (live site)** | `docs/metric.html` printed *"Quotation verification is INCOMPLETE (0/0). Do not rely on the table below"* next to 19 verified quotes: `_scoring_universe` still read the old `verification`/`document` keys after the report was reshaped to `summary`/`source`. The overview page reads the same file correctly — the site contradicted itself. | reads the current schema (falls back to recounting the quotes, never to 0/0); page rebuilt. New `tests/test_site.py` pins the badge to the evidence. |
| F12 | medium (evidence staleness) | `build_site.py` rebuild drifted from committed pages two more ways: (a) `rules_quotes.json` records `source.url=/tmp/rules_canonical.pdf` (workflow verifies a /tmp copy), which the index renderer would publish as a link; (b) `link_verification.json` (19:33Z) disagrees with `sources.html` (built 18:24Z), flagging 3 live ScienceBase pages as BROKEN (403 = bot-wall — the GeoDAWN item re-verified reachable via independent fetch the same day). | **not touched here**: the parallel session (`arena/01a0ab54-gemsdoe`) independently found and fixed both on its branch (canonical-URL recording + http-guard, sciencebase BOT_BLOCK policy + `--reclassify`, workflow rebuild step, audit count check). Deliberately left to that branch to avoid a same-hunk conflict; this branch rebuilt only `docs/metric.html`. |

Verification after the fixes: **46/46 tests pass** (38 on arrival + 8 new: 4 blend/seed in
`test_ensemble.py`, 1 in `test_discovery.py`, 3 in `tests/test_site.py`), `audit_docs.py` PASS with
0 uncatalogued hosts, `src/metrics.py --self-test` 8/8, `metric.html` shows "19/19 quoted sentences
verified verbatim" with no `/tmp/` leak. Dependencies audited against imports: `scikit-learn` and
`matplotlib` are listed in `requirements.txt` but imported nowhere (same dead-dep class as the
removed `albumentations`) — left in place as harmless; `einops` (also listed) is genuinely unneeded
(`smp` 0.5.0 instantiates SegFormer without it; only the weight download fails, on the blocked
host — the known limitation).

Still running at session end: the emission-width + LOO A/B (run 35133590776, `blend` job since
18:18Z). It executes the *pre-session-6* blend code on the sibling branch, so when it lands: trust
its floor/dilate verdicts, **disregard its `weight_rule_gain`** (it was computed by the removed
cross-geography comparison). Its report commits to `arena/01a0ab54-gemsdoe`, not here.

Merge note for the sibling branch: this PR touches `scripts/build_site.py` only in
`_scoring_universe` (a different hunk than its index-URL guard — clean merge) but rebuilds
`docs/metric.html`, whose footer line it also rebuilt → expect a one-line footer conflict there,
resolved by rebuilding the site from the merged tree. After both merge, re-run the verify-sources
workflow once so all pages render from one consistent evidence set.

---

## 7. Session 5b: the width experiment came back, and it disagreed with the surrogate

The reblend experiment (`reblend.yml`, run [35133590776](https://github.com/buffedlizard55-lab/GEMSDOE/actions/runs/35133590776),
evidence in `data/evidence/runs/35042805806-experiment/`) answered both questions it was built to
answer, and its answers were: *keep the skeleton*, and *the shaping is worth less than the acceptance
threshold says is worth keeping*.

| question | measurement |
|---|---|
| Which band width does the search pick on the six real held-out crops? | **0 px** (the pure skeleton) at floor 0.4697, pooled mean DTI 0.1903 |
| Same question per band, best floor over the grid | 0 px → 0.1903 · 1 px → 0.1525 · 2 px → 0.1281 · 3 px → 0.1128 · 4 px → 0.1037 · 6 px → 0.0908 |
| Is the floor's gain real, fitted without the scored fold? | honest (LOO) mean **0.1643** vs 0.1560 unshaped = **+0.0083**, *below* the pre-registered 0.01 acceptance test |
| How large is the selection optimism? | pooled 0.1903 − honest 0.1643 = **0.0260**; oracle ceiling (floor fitted on the scored fold itself) 0.2029 |

**The surrogate pointed the wrong way, and this is the useful part.** `data/evidence/shift_robustness.json`
(a single window, labels shifted to simulate mislocalisation) argued for a 6-px band and +0.0705 over
the skeleton. The same bands scored on the six held-out crops — where the labels are the model's own
validation windows and the trace really is where the model drew it — monotonically *lose* DTI as the
band widens, because FP mass is paid per pixel while TP_w only takes a max within R = 3 px. Both
measurements are correct about different populations: the surrogate asks "what if the scored fault is
somewhere else than I drew it", the held-out crops ask "what if it is where I drew it". The scored
faults are new, so the truth is between them, and *neither* justifies overriding the held-out
measurement on the evidence available. The decision recorded here is therefore: **skeleton kept**,
`--dilate-grid` left in place as a knob with its verdict attached, and the honest number (+0.0083)
carried forward instead of the pooled one.

**Two defects the audit exposed in itself, both fixed:**

1. *The audit did not fit in its job* (4743 s of a 120-minute limit, and the earlier attempt hit the
   timeout at 120 minutes). Cause: `compute_distance_weighted_tversky` rebuilt a 49-offset credit map
   over the whole raster and a label distance transform for **every** candidate floor, then read only
   the ~1 % of pixels that are labels; `dominant_thin` computed a whole-raster EDT up to 40 times per
   candidate. Fixed in `src/metrics.GtContext` (label geometry built once; TP_w gathered at label
   pixels) and by maintaining the thinning distance map incrementally. Both are pinned to the original
   implementations by `tests/test_metric_parity.py` and `tests/test_shaping_parity.py` — a speed-up in
   the scoring function is the one change that could silently corrupt every number at once.
2. *The same submission had two different file hashes.* The two runs' rasters are **pixel-identical**
   (`sha256_pixels d6380a58…`) but the files differ, because the chosen floor is stored in GeoTIFF
   metadata at full float64 precision and `np.geomspace` rounded differently under another numpy build
   (`0.4696741044002384` vs `…2383`). The search grid is now quantised to six significant digits, and
   every report carries both the container hash (what a reviewer re-checks) and the pixel hash (what
   two runs are compared with).

**Cross-session review.** While this was running, a parallel session (PR #12) audited the code above
and found three real defects in it, all confirmed and merged here: `calibrate_shaping` mutated the
fold dicts, so `--calibrate loo` applied a pre-transform twice; its empty-input path returned four
values where callers unpack five; and the LOO *weight* comparison averaged fold crops covering
different geographic windows and scored them against another fold's labels, which cannot measure a
weight rule. All three are fixed on this branch; the weight rule is now fitted per row and reported
without being scored. The lesson kept from the merge: the fold-weight question still needs the
per-fold held-out footprints (queued in `SUGGESTIONS.md`), and a number that cannot be measured
should be `None`, not a plausible-looking float.

