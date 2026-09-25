# `make-submission.yml` run evidence

One JSON per run of [`.github/workflows/make-submission.yml`](../../../.github/workflows/make-submission.yml)
— the route that builds the submission file without installing a model and without a DrivenData
login. The workflow writes this file itself (`Commit the measurement` step), so the directory is
empty until the first run; the page section that links here
([`docs/how_to_submit.html` §3 and §4](https://buffedlizard55-lab.github.io/GEMSDOE2/how_to_submit.html))
renders that fact instead of a placeholder.

What a run records — and only what it measured:

| key | meaning |
|---|---|
| `route`, `package` | which artifact was assembled (`adopted` / `baseline` / `both`) and which containers were written |
| `sha256` | hash of each emitted `.tif` / `.zip`, so a download can be checked against the run |
| `validator` | `scripts/validate_submission.py` exit status and its last line, for every emitted file |
| `generator` | the verdict of `scripts/check_site_generator.py`, i.e. whether the browser generator on the page still reproduces the artifact in this checkout |
| `payload_check` | `scripts/build_submission_payload.py --check`, the manifest pins vs the raster on disk |
| `size_guard` | bytes of each artifact, with the 100 KB stub threshold the workflow refuses below |
| `commit`, `run_url` | the SHA and the Actions URL the numbers came from |

**Rasters are never committed here.** The artifact is delivered as a workflow artifact
(`submission-run-<run_id>`, 90-day retention); a 569 KB GeoTIFF per run is how a repository drowns,
and the only rasters in Git are the pinned ones already under `data/evidence/`.

**A run in this directory is a format measurement, not a score.** The score exists only after a
human enrols, uploads the file, and the platform returns a public score — the gate that
`scripts/check_submission_readiness.py` calls `human_upload` and that no workflow in this repository
can close.
