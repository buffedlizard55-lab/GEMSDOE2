"""The proxy-eval workflow's multi-ensemble contract, executed rather than parsed.

WHY THIS FILE EXISTS
--------------------
The sweep job measures the shaping policy (floor x thinning x emission width) on the population the
prize actually scores: faults the labels do NOT contain.  Until session 13 it could only sweep ONE
Actions run's fold artifacts, i.e. one ensemble's mean.  But the field that gets submitted is the
mean over every live ensemble (11-17 folds over three seeds), and a floor optimum is a property of
the FIELD: the same floor that wins on a 6-fold mean is not evidence for a 15-fold mean.

`RUN_ID=a,b,c` closes that gap - and the failure modes are quiet, which is why they are tested here:
  * `cut -f2` returns the WHOLE line when there is no delimiter, so a single run id would silently
    become run_id_2 as well and the same folds would be blended TWICE (a double-count that looks
    like a bigger ensemble);
  * a directory that downloaded nothing could contribute silently, so a "multi-ensemble" sweep could
    be a one-ensemble sweep with a label on it;
  * the flags could reach the workflow but not the blend command.

Run: python -m pytest tests/test_proxy_eval_workflow.py -q
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "proxy-eval.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _render(script: str, **vals) -> str:
    mapping = {
        "github.run_id": "99999999999",
        "github.ref_name": "test-branch",
        "needs.params.outputs.run_id": str(vals.get("run_id", "35042805806")),
        "needs.params.outputs.run_id_2": str(vals.get("run_id_2", "")),
        "needs.params.outputs.run_id_3": str(vals.get("run_id_3", "")),
        "needs.params.outputs.sweep_label": str(vals.get("sweep_label", "")),
        "needs.params.outputs.dilate_grid": "0,1,2",
        "needs.params.outputs.shaping_grid": "11",
        "needs.params.outputs.shaping_values": "0,0.1",
        "needs.params.outputs.soft_band": "0",
        "needs.params.outputs.band_gammas": "1,2",
        "needs.params.outputs.submission": "data/evidence/runs/35042805806/submission.tif",
    }
    for key, value in mapping.items():
        script = script.replace("${{ " + key + " }}", value)
    script = re.sub(r"\$\{\{ ([^}|]+?) \|\| '([^']*)' \}\}", lambda m: m.group(2), script)
    script = re.sub(r"\$\{\{ [^}]*? \}\}", "", script)
    assert "$" + "{{" not in script, f"unrendered expression in: {script[:200]}"
    return script


def _run_params(tmp_path: Path, trigger_text: str) -> subprocess.CompletedProcess:
    script = _render(_workflow()["jobs"]["params"]["steps"][-1]["run"])
    (tmp_path / ".github" / "triggers").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".github" / "triggers" / "proxy-eval-params").write_text(trigger_text)
    out_file = tmp_path / "out"
    out_file.write_text("")
    env = dict(os.environ)
    env.update({"GITHUB_OUTPUT": str(out_file)})
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    r.outputs = dict(line.split("=", 1) for line in out_file.read_text().splitlines() if "=" in line)
    return r


def test_single_run_id_does_not_leak_into_the_second_slot(tmp_path):
    """`cut -f2` on a delimiter-free line returns the line itself: the silent double-count."""
    r = _run_params(tmp_path, "RUN_ID=35042805806\n")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert r.outputs["run_id"] == "35042805806"
    assert r.outputs["run_id_2"] == "" and r.outputs["run_id_3"] == ""


def test_comma_list_parses_into_three_slots(tmp_path):
    r = _run_params(tmp_path, "RUN_ID=35042805806,35249562910,35263581931\n")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert r.outputs["run_id"] == "35042805806"
    assert r.outputs["run_id_2"] == "35249562910"
    assert r.outputs["run_id_3"] == "35263581931"


def test_download_steps_are_guarded_and_land_in_their_own_directories():
    steps = _workflow()["jobs"]["sweep"]["steps"]
    downloads = [s for s in steps if (s.get("name") or "").startswith("Download fold artifacts")]
    assert len(downloads) == 3, "one download per ensemble slot"
    assert [s["with"]["path"] for s in downloads] == ["folds", "folds2", "folds3"]
    assert "run_id_2" in downloads[1]["if"] and "run_id_3" in downloads[2]["if"]
    assert "if" not in downloads[0], "the first run is always downloaded"


def _step(name_prefix: str) -> dict:
    for s in _workflow()["jobs"]["sweep"]["steps"]:
        if (s.get("name") or "").startswith(name_prefix):
            return s
    raise AssertionError(f"no sweep step starting with {name_prefix!r}")


def _fold_args_block(rendered: str) -> str:
    """The part of the sweep step that decides which fold directories reach the blend."""
    start = rendered.index("shopt -s nullglob")
    stop = rendered.index("python scripts/blend_submission.py")
    return rendered[start:stop] + 'printf "%s\\n" "${FOLD_ARGS[@]}"\n'


def test_every_downloaded_ensemble_reaches_the_blend(tmp_path):
    block_step = _step("Re-blend")
    assert "blend_submission.py" in block_step["run"], block_step.get("name")
    rendered = _render(block_step["run"])

    for base, folds in (("folds", [0, 1, 2, 3]), ("folds2", [6, 7]), ("folds3", [12])):
        for f in folds:
            (tmp_path / base / f"fold-{f}").mkdir(parents=True)
    (tmp_path / "folds4").mkdir()                          # an unmanaged directory must be ignored

    block = _fold_args_block(rendered)
    r = subprocess.run(["bash", "-c", block], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert "fold dirs passed to the blend: 7" in lines[0], r.stdout
    # order matters: folds, then folds2, then folds3 - the evidence dir of a blend names the first run
    assert lines[1:] == ["folds/fold-0/", "folds/fold-1/", "folds/fold-2/", "folds/fold-3/",
                         "folds2/fold-6/", "folds2/fold-7/", "folds3/fold-12/"], lines


def test_an_empty_download_cannot_pass_as_a_multi_ensemble_sweep(tmp_path):
    block = _fold_args_block(_render(_step("Re-blend")["run"]))
    r = subprocess.run(["bash", "-c", block], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode != 0, "a sweep with no fold artifacts must fail, not sweep nothing"
    assert "no fold artifacts downloaded" in (r.stdout + r.stderr)
