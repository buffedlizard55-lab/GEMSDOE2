#!/usr/bin/env python3
"""Live HTTP verification of every link in docs/data_catalog.csv — line by line.

The catalog carries human-written `verification_result` strings from earlier sessions.
Those are claims. This script replaces claims with *measurements*: it fetches every
`official_link` right now and records the status code, final URL after redirects,
content type, byte size and (for HTML) the page <title>.

Rules applied, so the output cannot flatter itself:
  * a login redirect is reported as LOGIN_REQUIRED, not as "verified"
  * a 4xx/5xx is reported as BROKEN
  * a redirect to a different host is reported with the destination so a silent
    takeover is visible
  * anything that times out is UNREACHABLE, never assumed good

Writes docs/link_verification.json and a rewritten docs/data_catalog.verified.csv whose
`verification_result` column is machine-generated. Run on a GitHub runner (the dev sandbox
reaches only github.com / pypi.org, so it would mark everything UNREACHABLE).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import urllib.error
import urllib.parse
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (compatible; gems-prize-link-audit/1.0; +https://github.com/buffedlizard55-lab/GEMSDOE)"
LOGIN_HINTS = ("/login", "/accounts/login", "signin", "sign-in", "/users/sign_in")

# Academic publishers and DOI resolvers routinely return 403 to any non-browser client
# (Cloudflare / bot mitigation). That is NOT evidence the link is broken, and reporting it
# as BROKEN would be its own small hallucination. These hosts are therefore classified
# separately as BOT_BLOCKED_<code> and excluded from the "problems needing review" count.
# The DOI itself remains the citable identifier regardless of what a scripted HEAD sees.
BOT_BLOCKING_HOSTS = (
    # sciencebase.gov added 2026-09-16 after a measured flip: the USGS ScienceBase host served
    # 200 to the *same* runner earlier in the day and then returned 403 to EVERY request from
    # it (all ten doi.org/10.5066/* rows and all three /catalog/item/* pages in one run), while
    # the identical content stayed reachable through the platform page fetcher the same day.
    # Several hosts behave like this under scripted load; the honest record is "the host
    # refused this client", with the raw status kept in the row, not "the link is broken".
    # Those rows are verified independently through their DOI landing pages - see
    # data/evidence/sciencebase_dois.json - and the flip itself is an irregularity to report,
    # not something to hide behind a silent retry.
    "sciencebase.gov", "doi.org", "onlinelibrary.wiley.com",
    "agupubs.onlinelibrary.wiley.com",
    "www.mdpi.com", "mdpi.com", "link.springer.com", "www.sciencedirect.com",
    "sciencedirect.com", "pubs.geoscienceworld.org", "academic.oup.com",
    "www.tandfonline.com", "tandfonline.com", "iopscience.iop.org",
)


# Result classes that are NOT a defect.  A 401 means credentials are required, a 403 from one of
# BOT_BLOCKING_HOSTS means the host refused this client, LOGIN_REQUIRED is the account-gated data
# tab - none of them says the resource is missing.  Anything else (BROKEN_*, UNREACHABLE,
# NOT_A_URL) is a problem a human should look at.
EXPECTED_OK = ("OK",)
EXPECTED_NON_OK = ("BOT_BLOCKED", "LOGIN_REQUIRED", "AUTH_REQUIRED")


def classify_row(result: str, status, url: str) -> str:
    """Map a recorded (result, status, url) onto the CURRENT policy.

    Used by --reclassify: when the policy changes (a host turns out to serve 403 to scripted
    clients, an endpoint turns out to need credentials) the measurements do not have to be thrown
    away and re-taken - the same rows are re-derived under the new rule, and the next full run
    re-measures them anyway.  Hand-editing a measurement is not an option; re-deriving it from the
    recorded status is.
    """
    if result.startswith("BROKEN_HTTP_"):
        code = int(str(result).rsplit("_", 1)[-1]) if str(result).rsplit("_", 1)[-1].isdigit() else status
        host = urllib.parse.urlparse(url).netloc.lower()
        if code == 401:
            return "AUTH_REQUIRED"
        if code in (401, 403, 429) and any(host == h or host.endswith("." + h)
                                           for h in BOT_BLOCKING_HOSTS):
            return f"BOT_BLOCKED_{code}"
    return result


def reseal(payload: dict) -> dict:
    """Recompute every derived field of a link-verification payload from its own rows."""
    results = payload.get("results", [])
    for r in results:
        r["result"] = classify_row(r.get("result", ""), r.get("status"), r.get("url", ""))
    counts: dict[str, int] = {}
    for r in results:
        k = r["result"].split("_TO_")[0]
        counts[k] = counts.get(k, 0) + 1
    problems = [r for r in results if not r["result"].startswith(EXPECTED_OK + EXPECTED_NON_OK)]
    expected_nonok = [r for r in results if r["result"].startswith(EXPECTED_NON_OK)]
    payload["summary_counts"] = counts
    payload["problems"] = problems
    payload["n_problems"] = len(problems)
    payload["expected_non_ok"] = expected_nonok
    payload["n_expected_non_ok"] = len(expected_nonok)
    payload["classification"] = {
        "expected_ok": list(EXPECTED_OK),
        "expected_non_ok": list(EXPECTED_NON_OK),
        "requires_review": "anything else: BROKEN_*, UNREACHABLE, NOT_A_URL",
        "note": ("A 401 means credentials are required and a 403 from a publisher or DOI resolver "
                 "means the host refused this scripted client; LOGIN_REQUIRED is the account-gated "
                 "data tab. None of those says the resource is missing."),
    }
    return payload


def check(url: str, timeout: int = 45) -> dict:
    out = {"url": url}
    if not url or not url.startswith("http"):
        out.update(status=None, result="NOT_A_URL")
        return out
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            final = r.geturl()
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip()
            body = b""
            if ctype.startswith("text/html"):
                body = r.read(200000)
            clen = r.headers.get("Content-Length")
            out.update(
                status=r.status,
                final_url=final,
                content_type=ctype,
                content_length=(int(clen) if clen and clen.isdigit() else None),
            )
            if body:
                m = re.search(rb"<title[^>]*>(.*?)</title>", body, re.S | re.I)
                if m:
                    t = re.sub(r"\s+", " ", m.group(1).decode("utf-8", "replace")).strip()
                    out["title"] = t[:200]
            same_host = urllib.parse.urlparse(final).netloc == urllib.parse.urlparse(url).netloc
            out["redirected"] = final.rstrip("/") != url.rstrip("/")
            out["redirected_offsite"] = not same_host
            low = final.lower()
            if any(h in low for h in LOGIN_HINTS):
                out["result"] = "LOGIN_REQUIRED"
            elif not same_host:
                out["result"] = f"OK_REDIRECTED_TO_{urllib.parse.urlparse(final).netloc}"
            else:
                out["result"] = f"OK_{r.status}"
    except urllib.error.HTTPError as e:
        host = urllib.parse.urlparse(url).netloc.lower()
        if e.code in (401, 403) and any(host == h or host.endswith("." + h)
                                        for h in BOT_BLOCKING_HOSTS):
            out.update(status=e.code, result=f"BOT_BLOCKED_{e.code}",
                       note="publisher/DOI resolver rejects scripted clients; not a broken link")
        elif e.code == 401:
            # A 401 means "credentials required", not "broken": e.g. api.github.com/search/code is
            # used here as a *verification method* row (it needs a token).  Reporting it as a broken
            # link would be a false positive in the repo's own audit table.
            out.update(status=e.code, result="AUTH_REQUIRED",
                       note="endpoint requires credentials; not a broken or missing resource")
        else:
            out.update(status=e.code, result=f"BROKEN_HTTP_{e.code}")
    except Exception as e:  # noqa: BLE001
        out.update(status=None, result="UNREACHABLE", error=repr(e)[:200])
    return out


def reached_count(counts: dict) -> int:
    """How many URLs got a real answer in a run, from its summary counts.

    `EXPECTED_OK` is a tuple of PREFIXES ("OK") while the keys are full result classes
    ("OK_200", "OK_REDIRECTED"), so this has to be a prefix test: `k in EXPECTED_OK` is False for
    every real key, which silently reads as "reached nothing" and would disable the guard below.
    """
    return sum(v for k, v in counts.items() if str(k).startswith(EXPECTED_OK))


def degraded_against(counts: dict, previous: dict | None,
                     by_url: dict | None = None) -> dict:
    """Would writing this run replace a measured record with an unreachable one?

    A restricted-egress environment marks every off-allowlist host UNREACHABLE, which is a claim
    about this machine rather than about the links.  The test is deliberately blunt - fewer than
    half the URLs the committed record reached - because the honest failure mode is "refuse and
    say so", not "guess which subset is real".
    """
    n_reached = reached_count(counts)
    n_before = reached_count((previous or {}).get("summary_counts", {}))
    was_answered = {str(r.get("url")): str(r.get("result"))
                    for r in (previous or {}).get("results", [])
                    if str(r.get("result", "")).startswith(EXPECTED_OK + EXPECTED_NON_OK)}
    flipped = sorted(u for u, res in was_answered.items()
                     if (by_url or {}).get(u, {}).get("result") == "UNREACHABLE")
    return dict(n_reached=n_reached, n_before=n_before, flipped=flipped,
                degraded=bool(n_before) and n_reached < 0.5 * n_before)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reclassify", nargs="?", const="docs/link_verification.json", default=None,
                    help="Recompute the derived fields of an existing --json-out file from its own "
                         "recorded statuses under the current policy, without re-fetching anything. "
                         "Use when the policy changes (a host started refusing scripted clients); the "
                         "next full run re-measures every URL. Default path: docs/link_verification.json")
    ap.add_argument("--catalog", default="docs/data_catalog.csv")
    ap.add_argument("--json-out", default="docs/link_verification.json")
    ap.add_argument("--csv-out", default="docs/data_catalog.verified.csv")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="write the outputs even though this run reached nothing (see the guard "
                         "below). Use only when the degraded record IS the intended record")
    a = ap.parse_args()

    if a.reclassify:
        path = Path(a.reclassify)
        payload = json.loads(path.read_text(encoding="utf-8"))
        before = payload.get("n_problems")
        payload = reseal(payload)
        payload["reclassified_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        payload["reclassified_note"] = (
            "Derived fields recomputed from the recorded per-URL statuses under the current policy "
            "(no re-fetch). The next full run re-measures every URL.")
        path.write_text(json.dumps(payload, indent=1))
        print(f"{path}: problems {before} -> {payload['n_problems']}; "
              f"counts {payload['summary_counts']}")
        return 0

    with open(a.catalog, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    urls = [r.get("official_link", "").strip() for r in rows]
    uniq = sorted({u for u in urls if u})
    print(f"{len(rows)} catalog rows, {len(uniq)} unique URLs")

    with ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(check, uniq))
    by_url = {r["url"]: r for r in results}

    counts: dict[str, int] = {}
    for r in results:
        k = r["result"].split("_TO_")[0]
        counts[k] = counts.get(k, 0) + 1
        print(f"  {r['result']:34s} {r['url'][:95]}")

    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # ---- refuse to replace a measured record with an unreachable one ----------------------
    # A sandbox with no egress produces an all-UNREACHABLE run whose every row would then read
    # as if the link had failed - which is a *worse* claim than the one it replaces, and a false
    # one: nothing was measured about the link, only about this machine's network. The committed
    # record is left alone unless the caller asks for the degraded one explicitly.
    prev = Path(a.json_out)
    before = None
    if prev.exists():
        try:
            before = json.loads(prev.read_text(encoding="utf-8"))
        except Exception:
            before = None
    d = degraded_against(counts, before, by_url)
    n_reached, n_before, flipped = d["n_reached"], d["n_before"], d["flipped"]
    # A wholesale flip to UNREACHABLE is a statement about THIS MACHINE, not about the links: the
    # sandbox reaches github.com and pypi.org and nothing else, so 77 of 84 URLs "fail" on a
    # perfectly healthy network of links. Writing that would replace a measured record with a false
    # one, and would do it under a `generated_by` line that used to claim a live-HTTP runner.
    if d["degraded"] and not a.allow_degraded:
        print(f"REFUSING to overwrite {prev}: this run reached {n_reached} URLs where the "
              f"committed record reached {n_before}, and {len(flipped)} URLs that previously "
              f"responded are now UNREACHABLE. That pattern describes restricted egress, not "
              f"broken links, so the measured record is left intact. Re-run where the network "
              f"works, or pass --allow-degraded to record the failure explicitly.")
        if flipped:
            print("  example flipped URLs: " + ", ".join(flipped[:5]))
        return 3
    env_note = (f"on a host with live HTTP ({n_reached} of {len(uniq)} URLs answered)"
                if n_reached else
                f"in an environment that reached NO URL at all (0 of {len(uniq)}; no egress)")
    # The classification policy lives at module level so it can be asserted in tests
    # (tests/test_links_classifier.py) instead of being re-derived by hand each time a host
    # changes its mind about scripted clients.
    problems = [r for r in results if not r["result"].startswith(EXPECTED_OK + EXPECTED_NON_OK)]
    expected_nonok = [r for r in results if r["result"].startswith(EXPECTED_NON_OK)]

    fields = list(rows[0].keys())
    for extra in ("verification_result", "verified_status", "verified_final_url", "verified_utc"):
        if extra not in fields:
            fields.append(extra)
    with open(a.csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            u = row.get("official_link", "").strip()
            res = by_url.get(u, {"result": "NO_LINK"})
            row["verification_result"] = res["result"]
            row["verified_status"] = res.get("status", "")
            row["verified_final_url"] = res.get("final_url", "")
            row["verified_utc"] = stamp
            w.writerow(row)

    payload = {
        "generated_utc": stamp,
        "generated_by": f"scripts/verify_links.py {env_note}",
        "n_urls_reached": int(n_reached),
        "catalog": a.catalog,
        "n_rows": len(rows),
        "n_unique_urls": len(uniq),
        "summary_counts": counts,
        "n_problems": len(problems),
        "problems": problems,
        "n_expected_non_ok": len(expected_nonok),
        "expected_non_ok": expected_nonok,
        "results": results,
        "classification": {
            "expected_ok": list(EXPECTED_OK),
            "expected_non_ok": list(EXPECTED_NON_OK),
            "requires_review": "anything else: BROKEN_*, UNREACHABLE, NOT_A_URL",
            "note": ("A 401 means credentials are required and a 403 from a publisher or DOI "
                     "resolver means the host refused this scripted client; LOGIN_REQUIRED is the "
                     "account-gated data tab. None of those says the resource is missing, so they "
                     "are recorded with their raw status and counted separately from problems."),
        },
        "note": (
            "Machine-measured. LOGIN_REQUIRED (DrivenData data tab) and BOT_BLOCKED_403/401 "
            "(publishers and doi.org rejecting scripted clients) are EXPECTED and are not "
            "counted as problems. Only BROKEN_* / UNREACHABLE / NOT_A_URL require review."
        ),
    }
    with open(a.json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nsummary: {json.dumps(counts)}")
    print(f"{len(problems)} problem link(s); wrote {a.json_out} and {a.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
