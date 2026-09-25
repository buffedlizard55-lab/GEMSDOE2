"""Evidence pushes must survive two jobs landing the same derived file at once.

Why: block-holdout.yml runs its folds as a matrix and a single trigger push can fire
pseudo-label.yml at the same moment (runs 35451858112 and 35451858126 both started from one merge
commit on 2026-09-19).  Every one of those jobs rewrites
``data/evidence/block_holdout/fold_gap_summary.json``, because each has just added a fold the summary
must count.  The previous sequence -- ``git commit; git pull --rebase; git push`` -- fails the job
when two of them interleave, and 300 minutes of runner evidence is lost with it.

``scripts/push_evidence.sh`` fixes that by *regenerating* the derived file from the merged tree
instead of resolving the conflict from either side.  These tests exercise it against real git
repositories: a bare origin and two clones that push concurrently, with a stub reader that derives
the summary from whichever fold-gap files the tree holds.  Nothing here touches GitHub.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "push_evidence.sh"
EVID = "data/evidence/block_holdout"

# stands in for scripts/read_landed_reports.py --gaps-only: the summary is a pure function of the
# fold-gap files present in the tree, which is exactly the property the helper relies on
STUB = """\
import json, pathlib
d = pathlib.Path("data/evidence/block_holdout")
gaps = sorted(d.glob("fold*_generalisation_gap.json"))
d.mkdir(parents=True, exist_ok=True)
(d / "fold_gap_summary.json").write_text(json.dumps(
    {"n_folds_committed": len(gaps), "folds": [g.name for g in gaps]}, indent=1))
"""


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r


def _helper(cwd: Path, msg: str, attempts: int = 4) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
               PYTHON=sys.executable, GITHUB_REF_NAME="main",
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    return subprocess.run(["bash", str(HELPER), msg, str(attempts)], cwd=str(cwd),
                          capture_output=True, text=True, env=env)


def _gap(fold: int) -> str:
    return json.dumps({"fold": fold, "candidate": "floor0.1_w0px",
                       "held_out_dti": 0.08, "held_out_ci95": [0.06, 0.10], "held_out_blocks": 8,
                       "trained_on_dti": 0.09, "trained_on_ci95": [0.07, 0.11],
                       "trained_on_blocks": 26, "generalisation_gap": 0.01, "reading": "synthetic"})


def _seed(clone: Path, summary: bool) -> None:
    (clone / "scripts").mkdir(parents=True, exist_ok=True)
    (clone / "scripts" / "read_landed_reports.py").write_text(STUB)
    (clone / EVID).mkdir(parents=True, exist_ok=True)
    if summary:
        (clone / EVID / "fold_gap_summary.json").write_text(
            json.dumps({"n_folds_committed": 0, "folds": []}))
    _git(clone, "add", "-A")
    _git(clone, "commit", "-m", "seed")


def _stage_fold(clone: Path, fold: int, regenerate: bool = True) -> None:
    (clone / EVID).mkdir(parents=True, exist_ok=True)   # git does not carry empty directories
    (clone / EVID / f"fold{fold}_generalisation_gap.json").write_text(_gap(fold))
    if regenerate:
        subprocess.run([sys.executable, "scripts/read_landed_reports.py", "--gaps-only", "--quiet"],
                       cwd=str(clone), check=True, capture_output=True, text=True)
    _git(clone, "add", "-f", f"{EVID}/fold{fold}_generalisation_gap.json",
         f"{EVID}/fold_gap_summary.json")


def _origin_tree(origin: Path, path: str) -> str:
    r = _git(origin, "show", f"main:{path}", check=False)
    return r.stdout if r.returncode == 0 else ""


@pytest.fixture()
def repos(tmp_path, request):
    """A bare origin plus two clones that will push the same derived file."""
    tracked_summary = getattr(request, "param", True)
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   check=True, capture_output=True, text=True)
    a = tmp_path / "jobA"
    subprocess.run(["git", "clone", str(origin), str(a)], check=True, capture_output=True, text=True)
    _git(a, "symbolic-ref", "HEAD", "refs/heads/main")
    _seed(a, tracked_summary)
    _git(a, "push", "-u", "origin", "main")
    # job B clones AFTER A's first push, i.e. it starts from the same tip A will race against
    b = tmp_path / "jobB"
    subprocess.run(["git", "clone", str(origin), str(b)], check=True, capture_output=True, text=True)
    return origin, a, b


def test_the_helper_parses_as_shell():
    r = subprocess.run(["bash", "-n", str(HELPER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_a_lonely_push_commits_and_pushes(repos):
    origin, a, _ = repos
    _stage_fold(a, 0)
    r = _helper(a, "evidence(block-holdout): fold 0 generalisation gap [skip ci]")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _origin_tree(origin, f"{EVID}/fold0_generalisation_gap.json"), "the fold gap did not land"
    summary = json.loads(_origin_tree(origin, f"{EVID}/fold_gap_summary.json"))
    assert summary["n_folds_committed"] == 1


def test_nothing_staged_is_not_a_failure(repos):
    origin, a, _ = repos
    before = _git(origin, "rev-list", "--count", "main").stdout.strip()
    r = _helper(a, "evidence(block-holdout): nothing [skip ci]")
    assert r.returncode == 0
    assert "nothing staged" in r.stdout
    assert _git(origin, "rev-list", "--count", "main").stdout.strip() == before


def test_a_conflicting_summary_is_regenerated_not_resolved(repos):
    """The case that lost evidence before: both jobs rewrite the tracked summary."""
    origin, a, b = repos
    _stage_fold(a, 0)
    assert _helper(a, "evidence(block-holdout): fold 0 [skip ci]").returncode == 0
    # job B stages BEFORE seeing job A's push, so its summary counts only its own fold
    _stage_fold(b, 1)
    r = _helper(b, "evidence(block-holdout): fold 1 [skip ci]")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "::warning::rebase blocked" in r.stdout, \
        "this test must actually hit the conflict path, or it proves nothing:\n" + r.stdout
    assert "regenerated" in r.stdout, "the summary must be re-derived from the merged tree"
    summary = json.loads(_origin_tree(origin, f"{EVID}/fold_gap_summary.json"))
    assert summary["n_folds_committed"] == 2, \
        f"the merged summary must count BOTH folds, got {summary}"
    assert _origin_tree(origin, f"{EVID}/fold0_generalisation_gap.json"), "job A's fold was lost"
    assert _origin_tree(origin, f"{EVID}/fold1_generalisation_gap.json"), "job B's fold was lost"


@pytest.fixture()
def untracked_summary_repos(tmp_path):
    """The same race, but no summary is committed yet - so it is untracked on both sides."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   check=True, capture_output=True, text=True)
    a = tmp_path / "jobA"
    subprocess.run(["git", "clone", str(origin), str(a)], check=True, capture_output=True, text=True)
    _git(a, "symbolic-ref", "HEAD", "refs/heads/main")
    _seed(a, summary=False)
    _git(a, "push", "-u", "origin", "main")
    b = tmp_path / "jobB"
    subprocess.run(["git", "clone", str(origin), str(b)], check=True, capture_output=True, text=True)
    return origin, a, b


def test_an_untracked_summary_collision_does_not_wedge_the_push(untracked_summary_repos):
    """git refuses to rebase over an untracked file the other side added - the helper must cope."""
    origin, a, b = untracked_summary_repos
    _stage_fold(a, 0)
    assert _helper(a, "evidence(block-holdout): fold 0 [skip ci]").returncode == 0
    _stage_fold(b, 1)
    r = _helper(b, "evidence(block-holdout): fold 1 [skip ci]")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "::warning::rebase blocked" in r.stdout, \
        "an added-on-both-sides file must reach the regeneration path:\n" + r.stdout
    summary = json.loads(_origin_tree(origin, f"{EVID}/fold_gap_summary.json"))
    assert summary["n_folds_committed"] == 2, summary
    assert _origin_tree(origin, f"{EVID}/fold1_generalisation_gap.json")


def test_a_retry_never_sweeps_up_a_file_it_was_not_given(repos):
    """The helper re-stages exactly what the caller staged - never a stray raster in the tree."""
    origin, a, b = repos
    _stage_fold(a, 0)
    assert _helper(a, "evidence(block-holdout): fold 0 [skip ci]").returncode == 0
    _stage_fold(b, 1)
    stray = b / EVID / "fold1_heldout.tif"          # 40 MB of float32 in the real tree
    stray.write_bytes(b"\0" * 1024)
    r = _helper(b, "evidence(block-holdout): fold 1 [skip ci]")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not _origin_tree(origin, f"{EVID}/fold1_heldout.tif"), \
        "an unstaged raster must never reach the remote"
    assert json.loads(_origin_tree(origin, f"{EVID}/fold_gap_summary.json"))["n_folds_committed"] == 2


def test_an_unreachable_remote_is_an_error_and_keeps_the_evidence(repos):
    """A 1 loses the commit but not the measurement, and says so as an error, not a warning."""
    origin, a, b = repos
    _stage_fold(b, 1)
    # point the clone at a remote that does not exist: every attempt must fail
    _git(b, "remote", "set-url", "origin", str(origin.parent / "no-such-remote.git"))
    r = _helper(b, "evidence(block-holdout): fold 1 [skip ci]", attempts=2)
    assert r.returncode == 1
    assert "::error::" in r.stdout
    assert (b / EVID / "fold1_generalisation_gap.json").exists(), \
        "giving up must not delete the evidence from the working tree"
