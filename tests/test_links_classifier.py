"""Status classification in scripts/verify_links.py — stdlib only.

WHY THIS EXISTS.  The catalog's verification table is only useful if "broken" means broken.  Two
false positives have already been found and fixed by hand:

  * api.github.com/search/code returns 401 (it needs a token; it is a *verification method* row,
    not a link) — reported as BROKEN_HTTP_401 until 2026-09-16;
  * USGS ScienceBase served 200 to a runner in the morning and then 403 to every request from the
    same runner in the afternoon — three of its item pages were reported as BROKEN_HTTP_403.

Both were real drift, and both were fixed by editing a host list.  These tests pin the behaviour so
the next such change is a decision rather than an accident: 401 is always "credentials required",
403/429 on a known bot-blocking host is "the host refused this client", and a 404 — which no
well-behaved page returns for a live resource — stays broken.
"""
import importlib.util
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("verify_links", ROOT / "scripts" / "verify_links.py")
verify_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_links)


def _http_error(url, code, body=b""):
    """Mimic urlopen's exception path for a given status code."""
    return urllib.error.HTTPError(url, code, "err", {}, None) if code else None


def _classify(monkeypatch_code, url):
    """Run verify_links.check(url) with urlopen forced to raise `monkeypatch_code`."""
    def fake_urlopen(req, timeout=None):                       # noqa: ARG001
        raise urllib.error.HTTPError(url, monkeypatch_code, "err", {}, None)

    real = verify_links.urllib.request.urlopen
    verify_links.urllib.request.urlopen = fake_urlopen
    try:
        return verify_links.check(url, timeout=1)
    finally:
        verify_links.urllib.request.urlopen = real


def test_401_is_credentials_not_broken():
    r = _classify(401, "https://api.github.com/search/code")
    assert r["result"] == "AUTH_REQUIRED", r
    assert r["status"] == 401
    assert "not a broken" in r["note"]


def test_403_on_a_bot_blocking_host_is_expected():
    for url in ("https://www.sciencebase.gov/catalog/item/589097b1e4b072a7ac0cae23",
                "https://doi.org/10.5066/P9Z6SA1Z",
                "https://www.sciencedirect.com/science/article/abs/pii/S0098300421000169"):
        r = _classify(403, url)
        assert r["result"] == "BOT_BLOCKED_403", (url, r)
        assert r["status"] == 403, "the raw status must stay in the record"


def test_403_on_an_unlisted_host_is_still_reported_as_broken():
    r = _classify(403, "https://example.invalid/private/thing")
    assert r["result"] == "BROKEN_HTTP_403", r


def test_404_is_broken_on_any_host():
    r = _classify(404, "https://www.sciencebase.gov/catalog/item/deadbeef")
    assert r["result"] == "BROKEN_HTTP_404", r


def test_only_expected_results_are_counted_as_expected():
    """The summary's classification policy must cover the non-broken non-OK classes and nothing else."""
    ok, non_ok = verify_links.EXPECTED_OK, verify_links.EXPECTED_NON_OK
    assert ok == ("OK",)
    assert set(non_ok) == {"BOT_BLOCKED", "LOGIN_REQUIRED", "AUTH_REQUIRED"}
    for r in ("BROKEN_HTTP_404", "UNREACHABLE", "NOT_A_URL"):
        assert not r.startswith(ok + non_ok), f"{r} must stay a problem"


def test_sciencebase_is_listed_as_bot_blocking():
    """Regression pin for the 2026-09-16 flip: this host must not be reported as broken."""
    assert "sciencebase.gov" in verify_links.BOT_BLOCKING_HOSTS


# ------------------------------------------------------------------ degraded-egress guard
def test_a_restricted_environment_must_not_overwrite_a_measured_record():
    """The sandbox reaches github.com and pypi.org and nothing else.

    A run there marks 77 of 84 catalog URLs UNREACHABLE - a statement about this machine, not about
    the links. Writing that over a record in which 72 URLs answered would replace a measurement
    with a falsehood, so `degraded_against` has to say so and `main()` has to refuse.
    """
    previous = dict(summary_counts={"OK_200": 61, "OK_REDIRECTED": 10, "OK_202": 1,
                                    "BOT_BLOCKED_403": 10, "AUTH_REQUIRED": 1, "LOGIN_REQUIRED": 1},
                    results=[dict(url="https://example.gov/a", result="OK_200"),
                             dict(url="https://example.gov/b", result="BOT_BLOCKED_403")])
    counts = {"UNREACHABLE": 77, "BROKEN_HTTP_400": 1, "OK_200": 6}
    by_url = {"https://example.gov/a": {"result": "UNREACHABLE"}}
    d = verify_links.degraded_against(counts, previous, by_url)
    assert d["n_reached"] == 6 and d["n_before"] == 72 and d["degraded"] is True
    assert d["flipped"] == ["https://example.gov/a"]


def test_the_same_record_is_not_degraded_by_a_healthy_run():
    previous = dict(summary_counts={"OK_200": 61}, results=[])
    counts = {"OK_200": 60, "UNREACHABLE": 4}
    assert verify_links.degraded_against(counts, previous, {})["degraded"] is False


def test_reached_count_treats_the_result_classes_as_prefixes():
    """`EXPECTED_OK` is ("OK",) while the keys are "OK_200"/"OK_REDIRECTED".

    A membership test there is always False, which would read as "reached nothing", disable the
    guard and mislabel the environment as having no egress.
    """
    assert verify_links.reached_count({"OK_200": 61, "OK_REDIRECTED": 10, "OK_202": 1,
                                       "UNREACHABLE": 12}) == 72
    assert verify_links.reached_count({"UNREACHABLE": 12}) == 0


def test_a_first_run_has_nothing_to_degrade():
    """With no committed record, the guard must not block the very first measurement."""
    d = verify_links.degraded_against({"UNREACHABLE": 84}, None, {})
    assert d["n_before"] == 0 and d["degraded"] is False
