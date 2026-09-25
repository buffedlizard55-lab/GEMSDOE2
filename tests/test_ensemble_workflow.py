"""The ensemble workflow's fold-identity contract, executed (not just parsed).

Why this file exists
--------------------
Actions run 35170395055 (2026-09-17, the queued SECOND 6-fold ensemble) had all six folds train
for ~3 hours and then threw every one of them away:

* the `Train fold` step overrode only ``training.mc_id=${{ matrix.fold }}`` (0..5), ignoring
  ``FOLD_OFFSET``, so ``src/train.py`` wrote ``heldout_mc0..5.npz`` / ``model_mc0..5_*.pt``;
* the `Stage fold outputs` step copied ``outputs/heldout_mc$((FOLD_OFFSET + matrix.fold)).npz``
  -> with ``FOLD_OFFSET=6`` that path never existed, ``cp`` failed, the step failed on every
  fold, `Upload fold artifact` was skipped (no ``if: always()``), and the blend job correctly
  refused to blend an incomplete fold set;
* the failure record could not even be committed, because the `Checkout` step of the blend job
  had been skipped (it had no ``if: always()``) and the ``if: always()`` evidence step then ran
  without a ``.git`` directory.

The previous test only checked that workflows *parse* (``tests/test_workflow_yaml.py``).  This
file goes further: it extracts the real ``run:`` shell from the workflow YAML and executes it
against planted artefacts that follow the naming contract of ``src/train.py``.  The positive test
plants ``heldout_mc6.npz`` with ``MC_ID=6``; the negative control plants ``heldout_mc0.npz`` and
requires the step to fail loudly.  A silent mismatch is the failure mode that cost 18 CPU-hours.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "train-ensemble.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _step(job: str, name: str) -> dict:
    for s in _workflow()["jobs"][job]["steps"]:
        if s.get("name") == name:
            return s
    raise AssertionError(f"no step named {name!r} in job {job!r}")


def _render(script: str, matrix_fold: int = 0) -> str:
    """Substitute the GitHub expression syntax the runner would expand."""
    return (script.replace("${{ matrix.fold }}", str(matrix_fold))
                  .replace("${{ needs.fold.result }}", "failure")
                  .replace("${{ github.run_id }}", "99999999999")
                  .replace("${{ github.sha }}", "deadbeef")
                  .replace("${{ github.ref_name }}", "test-branch"))


def _plant_outputs(tmp: Path, mc_in_manifest: int, heldout_mc: int) -> None:
    """Write exactly what src/train.py + src/inference.py write for a fold."""
    out = tmp / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prob_raw.tif").write_bytes(b"\x00" * 64)          # existence is what staging checks
    (out / "manifest.json").write_text(json.dumps(
        {"models": [{"file": f"model_mc{mc_in_manifest}_unetplusplus_resnet34.pt",
                     "arch": "unetplusplus", "encoder": "resnet34", "mc": mc_in_manifest, "dti": 0.1}]}))
    np.savez_compressed(out / f"heldout_mc{heldout_mc}.npz",
                        pred=np.zeros((4, 4), np.float16), gt=np.zeros((4, 4), np.uint8))
    (out / f"model_mc{mc_in_manifest}_unetplusplus_resnet34.pt").write_bytes(b"\x00" * 128)
    (out / "train.log").write_text("train\n")
    (out / "inference.log").write_text("inference\n")


def _run_staging(tmp: Path, env_extra: dict, matrix_fold: int = 0) -> subprocess.CompletedProcess:
    import sys

    script = _render(_step("fold", "Stage fold outputs")["run"], matrix_fold=matrix_fold)
    env = dict(os.environ)
    # The staging script calls a bare `python`, so it inherits whatever is first on PATH.  On a
    # runner that is the interpreter `pip install -r requirements.txt` populated; in a local
    # venv it is the SYSTEM python, which has no numpy, and the step then failed for an
    # environment reason instead of testing the contract (measured 2026-09-18: two red tests
    # whose only output was "ModuleNotFoundError: No module named 'numpy'").  Pinning the
    # interpreter's own bin directory to the front of PATH makes the executed contract test
    # hermetic without changing the script under test.
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env.update(env_extra)
    return subprocess.run(["bash", "-c", script], cwd=tmp, env=env,
                          capture_output=True, text=True)


# ---------------------------------------------------------------- the executed contract
def test_staging_accepts_the_trainer_naming_with_an_offset(tmp_path):
    """FOLD_OFFSET=6 + matrix.fold=0 -> the trainer's mc id is 6; staging must find mc6."""
    _plant_outputs(tmp_path, mc_in_manifest=6, heldout_mc=6)
    r = _run_staging(tmp_path, {"MC_ID": "6", "FOLD_OFFSET": "6", "ENSEMBLE_SEED": "43",
                                "GITHUB_SHA": "deadbeef"})
    assert r.returncode == 0, f"staging failed:\n{r.stdout}\n{r.stderr}"
    fold = tmp_path / "fold"
    names = sorted(p.name for p in fold.iterdir())
    assert "prob_raw.tif" in names and "manifest.json" in names
    assert "heldout_mc6.npz" in names, names
    assert any(n.startswith("model_mc6_") for n in names), names
    params = json.loads((fold / "fold_params.json").read_text())
    assert params["mc_id"] == 6 and params["fold_offset"] == 6 and params["ensemble_seed"] == 43


def test_staging_refuses_a_fold_whose_identity_does_not_match(tmp_path):
    """The exact run-35170395055 defect: job says mc6, the trainer wrote mc0.

    Staging must fail loudly instead of producing a fold the blend job cannot calibrate on.
    """
    _plant_outputs(tmp_path, mc_in_manifest=0, heldout_mc=0)
    r = _run_staging(tmp_path, {"MC_ID": "6", "FOLD_OFFSET": "6", "ENSEMBLE_SEED": "43",
                                "GITHUB_SHA": "deadbeef"})
    assert r.returncode != 0, "a fold identity mismatch must fail the step"
    assert "mc ids [0] != workflow fold identity 6" in (r.stdout + r.stderr)


def test_staging_refuses_a_missing_heldout_crop(tmp_path):
    """No calibration crop -> the blend job's pooled shaping search is impossible; fail here."""
    out = tmp_path / "outputs"
    out.mkdir(parents=True)
    (out / "prob_raw.tif").write_bytes(b"\x00" * 64)
    (out / "manifest.json").write_text(json.dumps({"models": [{"mc": 6}]}))
    r = _run_staging(tmp_path, {"MC_ID": "6", "FOLD_OFFSET": "6", "ENSEMBLE_SEED": "43",
                                "GITHUB_SHA": "deadbeef"})
    assert r.returncode != 0
    assert "no outputs/heldout_mc*.npz" in (r.stdout + r.stderr)


# ---------------------------------------------------------------- the parameters step
def test_parameters_step_computes_the_offset_fold_identity(tmp_path):
    (tmp_path / ".github" / "triggers").mkdir(parents=True)
    (tmp_path / ".github" / "triggers" / "ensemble-params").write_text(
        "# a comment\nFOLD_OFFSET=6\nENSEMBLE_SEED=43\n")
    env_file = tmp_path / "github_env"
    env_file.touch()
    import os
    script = _render(_step("fold", "Read ensemble parameters (committed trigger file overrides the defaults)")["run"],
                     matrix_fold=2)
    env = dict(os.environ)
    env["GITHUB_ENV"] = str(env_file)
    env["FOLD_OFFSET"] = "0"      # stale job-level default: must NOT win over the trigger file
    env["ENSEMBLE_SEED"] = "42"
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    written = dict(line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line)
    assert written["FOLD_OFFSET"] == "6", written
    assert written["ENSEMBLE_SEED"] == "43", written
    assert written["MC_ID"] == "8", written                       # 6 + matrix.fold(2)


def test_train_step_applies_the_workflow_fold_identity():
    run = _step("fold", "Train fold ${{ matrix.fold }}")["run"]
    assert 'training.mc_id="$MC_ID"' in run, run
    assert 'training.seed="$ENSEMBLE_SEED"' in run, run
    # the raw matrix index must not be used as the fold identity anywhere in the trainer call
    assert "training.mc_id=${{ matrix.fold }} " not in run


def test_no_step_hardcodes_a_heldout_file_name():
    """Staging must glob the trainer's output; a composed name is what broke run 35170395055."""
    run = _step("fold", "Stage fold outputs")["run"]
    assert "heldout_mc$(( FOLD_OFFSET" not in run
    assert "heldout_mc*.npz" in run


# ---------------------------------------------------------------- recoverability of a failure
def test_fold_artifact_upload_runs_even_when_staging_fails():
    """Three hours of training must survive a staging hiccup: `if: always()` on the upload."""
    up = _step("fold", "Upload fold artifact")
    assert up.get("if") == "always()", up.get("if")
    assert up["with"]["path"] == "fold/"


def test_fold_diagnostics_artifact_cannot_be_mistaken_for_a_fold():
    """A crashed trainer must leave a log artifact - named so the `fold-*` download never sees it."""
    steps = _workflow()["jobs"]["fold"]["steps"]
    diag = [s for s in steps if str(s.get("name", "")).startswith("Upload fold diagnostics")]
    assert diag, [s.get("name") for s in steps]
    name = diag[0]["with"]["name"]
    assert "foldlogs-" in name and not name.startswith("fold-"), name
    assert diag[0]["with"]["path"] == "diag/"
    keep = [s for s in steps if str(s.get("name", "")).startswith("Keep the trainer")]
    assert keep and keep[0].get("if") == "failure()", keep


def test_blend_job_checkout_runs_first_and_always():
    """A failed ensemble must still be able to commit FAILED.json (needs a .git directory)."""
    steps = _workflow()["jobs"]["blend"]["steps"]
    assert steps[0].get("name") == "Checkout", [s.get("name") for s in steps]
    assert steps[0].get("if") == "always()", steps[0].get("if")


def test_blend_is_gated_on_usable_folds_not_on_the_job_result(tmp_path):
    """Run 35249562910: five folds trained for 3 h, one failed, and the binary rule discarded all five.

    The gate must count fold directories that actually carry a probability raster AND a held-out
    calibration crop, blend at or above MIN_FOLDS, and refuse below it (or with none at all).
    """
    script = _render(_step("blend", "Fold inventory + minimum-fold gate")["run"])
    env_file = tmp_path / "github_env"
    env_file.touch()
    import os

    def run(n_good: int, n_bad: int = 0, min_folds: str = "4"):
        import shutil
        shutil.rmtree(tmp_path / "folds", ignore_errors=True)
        (tmp_path / "folds").mkdir()
        for i in range(n_good):
            d = tmp_path / "folds" / f"fold-{i}"
            d.mkdir()
            (d / "prob_raw.tif").write_bytes(b"x")
            np.savez_compressed(d / f"heldout_mc{i}.npz", pred=np.zeros((2, 2)), gt=np.zeros((2, 2)))
        for i in range(n_bad):
            d = tmp_path / "folds" / f"fold-{90 + i}"
            d.mkdir()
            (d / "train.log").write_text("crashed\n")          # logs only, no raster
        env = dict(os.environ)
        env.update({"GITHUB_ENV": str(env_file), "MIN_FOLDS": min_folds})
        env.pop("FOLD_COUNT", None)
        env_file.write_text("")
        r = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env,
                           capture_output=True, text=True)
        written = dict(line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line)
        return r, written

    r, w = run(n_good=6)
    assert r.returncode == 0 and w["FOLD_COUNT"] == "6", (r.stdout, r.stderr)

    r, w = run(n_good=5, n_bad=1)          # exactly the run-35249562910 shape
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert w["FOLD_COUNT"] == "5"
    assert "blending 5 of 6 folds" in r.stdout

    r, w = run(n_good=4)                   # at the floor: allowed, still recorded
    assert r.returncode == 0 and w["FOLD_COUNT"] == "4"

    r, _ = run(n_good=3)                   # below the floor: refuse loudly
    assert r.returncode != 0
    assert "below MIN_FOLDS" in (r.stdout + r.stderr)

    r, _ = run(n_good=0)                   # nothing usable at all
    assert r.returncode != 0
    assert "no usable fold artifacts" in (r.stdout + r.stderr)

    # the blend step must consume the inventory, not a raw glob (an unusable dir would crash it)
    blend_run = _step("blend", "Blend + shape + write submission")["run"]
    assert "--folds $(cat usable_folds.txt)" in blend_run, blend_run


def test_blend_commit_step_pushes_with_retries_and_a_hard_failure():
    run = _step("blend", "Commit run report + submission to the branch")["run"]
    assert "fold_params.json" in run, "per-fold identity must be committed with the run record"
    assert run.count("git push") >= 1
    assert 'pushed="yes"' in run and "FATAL" in run, "an unpushable record must fail the step"


# ---------------------------------------------------------------------------------------------
# The shipped emission width is a POLICY parameter, read from the committed trigger file.
#
# `--min-dilate` exists because the pooled in-domain calibration always returns the narrowest band
# (MEASURED: 0.1903 -> 0.0908 held-out DTI from a skeleton to a 6 px band on the catalogue), while
# both prize phases score faults that are NOT in the training labels.  A measured width can only
# reach a run if the workflow reads it, so the read is executed here rather than assumed.
# ---------------------------------------------------------------------------------------------

def test_blend_reads_min_dilate_from_the_committed_parameter_file(tmp_path):
    step = _step("blend", "Read the shipped emission width (committed parameter file)")
    script = _render(step["run"])
    assert ".github/triggers/ensemble-params" in script

    (tmp_path / ".github" / "triggers").mkdir(parents=True)
    env_file = tmp_path / "env"
    env_file.write_text("")
    out_file = tmp_path / "out"
    env = dict(os.environ)
    env.update({"GITHUB_OUTPUT": str(out_file)})

    def run(params_text: str | None) -> str:
        p = tmp_path / ".github" / "triggers" / "ensemble-params"
        if params_text is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(params_text)
        out_file.write_text("")
        r = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout, r.stderr)
        written = dict(line.split("=", 1) for line in out_file.read_text().splitlines() if "=" in line)
        return written["min_dilate"]

    assert run(None) == "0", "no parameter file must mean the unchanged behaviour, not a crash"
    assert run("FOLD_OFFSET=6\nENSEMBLE_SEED=43\n") == "0"
    assert run("MIN_DILATE=16\n") == "16"
    # comments must not be parsed as values (the file is heavily commented)
    assert run("# MIN_DILATE=16 is the decided width\nMIN_DILATE=8\n") == "8"


def test_blend_passes_min_dilate_to_the_blender():
    step = _step("blend", "Blend + shape + write submission")
    assert "--min-dilate" in step["run"], "the measured width never reaching the blender would ship 0"
    assert "steps.shape.outputs.min_dilate" in step["run"]
