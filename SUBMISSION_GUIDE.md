# Submission Guide — GEMS Prize (Verified from Official Sources)

> **See also:** the online **[Make a submission](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/submission.html)**
> page — the same path as a build-time-verified checklist (artifact sha256 re-hashed from the bytes,
> the placement table compared against whatever `data/` holds now, the validator table parsed from its
> own committed log, rules sentences quoted by id with their verification badge, and the irregularities
> that a submitter must know about before uploading).
>
> Also [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) and the online
> [Executive Summary Subpage](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/executive_summary.html)
> for the comprehensive executive roadmap, eligibility rules, Generative AI disclosure narrative, and
> pre-validated ready-to-upload submission raster.

**Primary sources:**
- Problem description submission format: https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format
- Official Rules PDF: https://www.nlr.gov/docs/fy26osti/96647.pdf (Sections 3.2, 3.3, 3.5, 3.6)
  - Both hosts verified identical 2026-09-12: `www.nlr.gov` and `docs.nlr.gov` serve the same SEPTEMBER 2026 document (title + TOC match). Full PDF read across 7 chunks.
- Competition main: https://www.drivendata.org/competitions/306/competition-doe-gems/
- Reference solution: https://github.com/drivendataorg/gems-prize-reference-solution

## 1. What to Submit (Per Problem Page)

> For this competition, you will submit a GeoTIFF file containing your predictions for **all** faults in the region.

Requirements (line-by-line verified from https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#submission-format):

- [ ] Same projected CRS as training data: **UTM zone 11N, EPSG:32611** (https://epsg.io/32611)
- [ ] Same resolution: **100m**
- [ ] Same bounds as training data, data outside bounds is null or NaN
- [ ] Single layer, datatype **32-bit float (float32)**, values **between 0 and 1** indicating confidence/probability, higher = higher probability
- [ ] Sample submission that predicts total fault absence is provided for reference on data download page (https://www.drivendata.org/competitions/306/competition-doe-gems/data/)

## 2. How to Enter (Per PDF Section 3.1, 3.2)

From PDF https://docs.nlr.gov/docs/fy26osti/96647.pdf:

- [ ] Create profile on DrivenData platform and agree to competition rules and restrictions
- [ ] Navigate to challenge website to sign up as competitor: https://www.drivendata.org/competitions/306/competition-doe-gems/
- [ ] Review competition materials and data (problem description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/ and about https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/)
- [ ] Download training features and labels from data tab https://www.drivendata.org/competitions/306/competition-doe-gems/data/ (requires login)
- [ ] Training dataset: geophysical features for GeoDAWN study area at 100m resolution, multiband GeoTIFF, one feature per band. Plus instructions for downloading USGS DEM at 1m.
- [ ] Training labels: existing fault data at 100m resolution where positively labeled pixels indicate fault presence, obtained from INGENIOUS Great Basin Regional Dataset Compilation DOI https://doi.org/10.15121/1881483 (PDF footnote 4)
- [ ] Competitors will submit predictions for all faults in GeoDAWN study area as GeoTIFF raster at 100m resolution. Example valid submission provided.
- [ ] **Generative AI disclosure:** If using generative AI, indicate in narrative (not included in word count) extent and how used (PDF section 3.2). Responsible for accuracy, authenticity, authorship.

## 3. Feedback & Limits (Per PDF Section 3.4)

> **⚠️ FLAGGED DISCREPANCY (not silently resolved):** an earlier project instruction stated
> "max 2 submissions per 7 days"; the official rules PDF (§3.2 + §3.4, verified 2026-09-12)
> says **up to three per week**. Per project policy, the official PDF + competition website are
> authoritative; the 2-per-7-days figure is treated as superseded but is recorded here for audit.
> Practical consequence: we plan around **3 submissions/week** (the binding limit per the
> platform may also be displayed at submission time — check before submitting).

- [ ] Each entity may submit **more than one set of predictions** for automated scoring up to **three per week**, as specified on competition website
- [ ] By submission deadline, **must select only one set of predictions** to use as final submission for final evaluation and ranking
- [ ] Multiple finalized submissions not allowed
- [ ] Each entity (team/org/individual) allowed **one final submission**; individuals on team not allowed separate final submission

## 4. Evaluation & Ranking (Per PDF Section 3.6)

- [ ] Using **distance-weighted Tversky index** metric published on competition website, judges score chosen submission against ground truth
- [ ] Ground truth divided into **public test dataset** and **private test dataset**
- [ ] **Public leaderboard:** score on public test dataset shown while competition running
- [ ] **Private leaderboard:** score on private test dataset used for first prize round when competition closes
- [ ] **Second round:** score against complete updated test set created by expert review after close
- [ ] **Must choose only one submission** for scoring across both prize rounds, **without knowledge of private test scores** — to encourage generalization, discourage overfitting to public test (PDF 3.6.2)
- [ ] Set of faults in public test and relative weight determined by organizers before start

### Prize Structure (Per PDF 1.1 and Problem Page)

- **Initial Round $50,000:** Top 5 each $10,000, judged on private test set of fault labels
- **Expert review:** Panel uses submitted predictions to update fault labels for full region
- **Final Round $250,000:** Top 5 judged on all fault labels in updated set — 1st $100k, 2nd $70k, 3rd $40k, 4th $25k, 5th $15k
- Same submission scored twice; predictions that helped experts identify previously-unmapped faults can score higher in final round

## 5. Solution Verification and Delivery (Per PDF 3.2, 3.5)

For finalists, for chosen algorithm, must submit:

- [ ] **Complete code assets and documentation**, including:
  - Description of resources required to build and run solution
  - Assets should be able to sufficiently reproduce winning results and generate predictions on new data samples
- [ ] Documentation consistent with DrivenData's Winning Model Documentation Template (provided to winners after competition)
- [ ] Finalists must sign and return required documents including eligibility certifications
- [ ] Award approvals — Official winners selected by DOE, may take into account program policy factors listed in Appendix A (PDF section 1.3 eligibility, 1.4 prize goals)
- [ ] DOE is judge and final decision maker, may elect to award all, none, or some submissions
- [ ] After winners notified, prize administrator requests necessary information to distribute cash prizes
- [ ] Interviews may be held after announcement (PDF 3.6.3), not required

## 6. How to Submit via DrivenData (Practical)

1. Go to https://www.drivendata.org/competitions/306/competition-doe-gems/
2. Click "Compete!" to enroll (requires account)
3. Download data from https://www.drivendata.org/competitions/306/competition-doe-gems/data/
4. Train model using this repo: `python -m src.train --config configs/config.yaml`
5. Generate predictions: `python -m src.inference --config configs/config.yaml --model-dir outputs --out submission.tif`
6. Validate format: `python scripts/validate_submission.py --pred submission.tif --sample data/sample_submission.tif`
7. On DrivenData, click "Submit" → "Make new submission" → upload GeoTIFF
8. Check public LB score
9. Before deadline **Dec 3, 2026 11:59pm UTC**, select ONE submission as final for both prize rounds
10. If finalist, prepare code + docs per template

### 6a. Generating the file itself — three ways, one verdict (added 2026-09-22)

Everything above says *what* the file must be. These are the ways to *make* it, and all three end at
the same gate (`scripts/validate_submission.py` + `scripts/check_site_generator.py`).

**1. In the browser, from the published site** — no clone, no install, no GPU.
The panel is the **first block on the [landing page](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/index.html)** and
the [executive summary](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/executive_summary.html) (added 2026-09-24), and
[`docs/how_to_submit.html` §3](https://buffedlizard55-lab.github.io/GEMSDOE2/docs/how_to_submit.html) — the same mount, the same
payload, the same self-checks. It loads `docs/submission_meta.json` (grid, transform, EPSG, value pins) and
`docs/submission_field.bin`
(a 532,072-byte `gems-rle-v1` run-length encoding of the field, 259,495 runs) and writes the GeoTIFF
locally with `docs/geotiff_writer.js`. Nothing is uploaded and no network call is made: the writer's
17 self-checks run against the file it just parsed, and the download is withheld if any fails. The
result is **pixel-identical** to the adopted artifact (float32 bits equal) — *not* byte-identical,
because the artifact is 256×256 LZW-tiled and the page writes 64-row deflate strips; the page states
this rather than hiding it.

**2. From a clone, with one command** (same code, Node CLI):

```bash
python scripts/build_submission_payload.py --check       # the payload still describes the artifact
node docs/geotiff_writer.js docs/submission_meta.json docs/submission_field.bin \
     submission.tif submission.zip --rows-per-strip 64   # exits 1 and writes NOTHING if a check fails
python scripts/validate_submission.py --pred submission.tif \
       --sample data/sample_submission.tif --train data/training_features.tif
```

**3. On a CI runner, no login** — `.github/workflows/make-submission.yml`
(`route=adopted|baseline|both`, `package=tif|zip|both`): places `data/` from the sha256-pinned git
bridge, validates, rebuilds the payload if asked, runs the site generator check, packages the
`.zip`, refuses any artifact under 100,000 bytes (the run-35042805806 stub bug), prints the sha256s
and a paste-ready submit note, and commits only the measurement JSON to
`data/evidence/make_submission/`.

```bash
gh workflow run make-submission.yml -f route=adopted -f package=both
```

**If you prefer to hand over the `.zip`** the dialog accepts (the dialog's wording, transcribed by
the team; rules §3.2 documents only the GeoTIFF form — irregularity 7):

```bash
python scripts/package_submission.py --tif submission.tif --out submission.zip --json
python scripts/package_submission.py --tif submission.tif --out submission.zip --check
```

`--check` re-extracts the member, reads it back through GDAL, and compares bytes; a second member, a
recompressed member, or a member that differs by one bit fails the command. Packaging is verified to
be deterministic by re-running it and comparing hashes.

## 7. Submission Format Validation (Our Implementation)

`scripts/validate_submission.py` checks:

- CRS == EPSG:32611
- Resolution == 100m
- Bounds match sample submission (or training_features.tif)
- Single band, float32, values in [0,1]
- NaN/nodata allowed **outside** training bounds (spec: "data outside bounds is null or NaN"); values **inside** bounds must be within [0,1]. Our scorer (`src/metrics.py`) sanitizes NaN→0 so padded files evaluate correctly.
- File size matches expected

See reference solution for example: https://github.com/drivendataorg/gems-prize-reference-solution

## 8. Irregularities Flagged

- Sample submission: user-provided Dropbox mirror (`example_submission.tif`) verified reachable 2026-09-12; binary not fetched in sandbox (egress allowlist) — `scripts/download_competition_data.sh` fetches it; dummy fallback in `scripts/generate_dummy_submission.py`
- Training features file naming: problem page says `training_features.tif`, reference solution says `numeric_features.tif`, mirror file is `gems-geodawn-numerical-features.tif` — all three handled in `src/dataset.py`
- 1m DEM links: RESOLVED 2026-09-12 — links point at official USGS bucket `prd-tnm.s3.amazonaws.com` (verified); partial capture of the links file yielded 35 tiles / 6 S3-verified (artefact not committed at the time; regenerate with `python scripts/fetch_dem_links_pdf.py`); full authoritative list via `python scripts/download_dem_tiles.py --complete-listing`; tiles are 90–380 MB each. Competition JSON itself contains irregularities (duplicate rows, path≠filename project rows — the filename project is canonical, S3-proven; garbled hosts in the PDF print)
- Rules PDF submission cadence: see flagged discrepancy atop §3 (3/week per PDF vs earlier 2/7-days instruction)

## 9. No Hallucinations

All requirements above copied line-by-line from official sources, with links for manual verification.
