#!/usr/bin/env python3
"""Reassemble the official competition rasters from the git bridge in data/bridge/.

Receiving side of scripts/make_data_bridge.py.  Verifies, in this order:

1. every part/blob sha256 + byte size against data/bridge/manifest.json;
2. the concatenated feature stack's whole-file sha256 against the manifest;
3. the manifest pins against the independent inventory
   data/evidence/inventory.json when that file is present (belt and braces).

Only then does it write the canonical names the pipeline expects:

    data/training_features.tif   (418,912,844 B - 19-band feature stack)
    data/labels.tif              (  425,830 B   - existing faults, binary band 1)
    data/sample_submission.tif   (1,599,597 B   - submission template)

Idempotent: files whose sha256 already matches are left untouched.  No manual
input; any mismatch aborts before anything is written.

Usage:
    python scripts/assemble_data_bridge.py            # default data/bridge -> data/
    python scripts/assemble_data_bridge.py --verify   # verify only, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

INVENTORY = Path("data/evidence/inventory.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def place(src: Path, dst: Path, sha_expect: str, verify_only: bool) -> None:
    if dst.exists() and sha256_file(dst) == sha_expect:
        print(f"OK   {dst} already in place with the pinned sha256 (untouched)")
        return
    if verify_only:
        print(f"WOULD place {dst} from {src}")
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copyfile(src, tmp)
    if sha256_file(tmp) != sha_expect:
        tmp.unlink(missing_ok=True)
        sys.exit(f"FAIL: reassembled {dst} does not match sha256 {sha_expect}")
    tmp.replace(dst)
    print(f"OK   wrote {dst} ({dst.stat().st_size} B, sha256 verified)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bridge", default="data/bridge")
    ap.add_argument("--out", default="data")
    ap.add_argument("--verify", action="store_true",
                    help="verify only; do not write anything")
    args = ap.parse_args()
    bridge, out = Path(args.bridge), Path(args.out)

    manifest_path = bridge / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"FAIL: {manifest_path} not found. The bridge is committed by "
                 ".github/workflows/place-competition-data.yml; pull the branch first.")
    manifest = json.loads(manifest_path.read_text())

    pins = {}
    if INVENTORY.exists():
        inv = json.loads(INVENTORY.read_text())
        pins = {f["name"]: f for f in inv["files"]}

    failures = 0
    for entry in manifest["files"]:
        name, sha_expect = entry["name"], entry["sha256"]
        # cross-check the manifest against the independent inventory when available
        if name in pins and (pins[name]["sha256"] != sha_expect
                             or pins[name]["bytes"] != entry["bytes"]):
            print(f"FAIL {name}: manifest pin {entry['bytes']}B/{sha_expect[:12]}… != "
                  f"inventory pin {pins[name]['bytes']}B/{pins[name]['sha256'][:12]}…")
            failures += 1

        if "parts" in entry:
            # verify every part first; only then concatenate
            for p in entry["parts"]:
                pp = bridge / p["name"]
                if not pp.exists():
                    print(f"FAIL missing part {pp}")
                    failures += 1
                    continue
                got = sha256_file(pp)
                if got != p["sha256"] or pp.stat().st_size != p["bytes"]:
                    print(f"FAIL part {p['name']}: {pp.stat().st_size} B {got[:12]}… "
                          f"!= pinned {p['bytes']} B {p['sha256'][:12]}…")
                    failures += 1
            if failures:
                continue
            joined = out / name
            if not (joined.exists() and sha256_file(joined) == sha_expect):
                if args.verify:
                    print(f"WOULD concatenate {len(entry['parts'])} parts -> {joined}")
                else:
                    out.mkdir(parents=True, exist_ok=True)
                    tmp = joined.with_suffix(joined.suffix + ".part")
                    h = hashlib.sha256()
                    with tmp.open("wb") as wf:
                        for p in entry["parts"]:
                            with (bridge / p["name"]).open("rb") as rf:
                                for chunk in iter(lambda: rf.read(1 << 20), b""):
                                    h.update(chunk)
                                    wf.write(chunk)
                    if h.hexdigest() != sha_expect:
                        tmp.unlink(missing_ok=True)
                        print(f"FAIL {name}: concatenated sha256 {h.hexdigest()[:12]}… "
                              f"!= pinned {sha_expect[:12]}…")
                        failures += 1
                    else:
                        tmp.replace(joined)
                        print(f"OK   concatenated {len(entry['parts'])} parts -> {joined} "
                              f"({joined.stat().st_size} B, sha256 verified)")
            else:
                print(f"OK   {joined} already in place with the pinned sha256")
            place(out / name, out / entry["canonical"], sha_expect, args.verify)
        else:
            src = bridge / name
            if not src.exists():
                print(f"FAIL missing {src}")
                failures += 1
                continue
            got = sha256_file(src)
            if got != sha_expect or src.stat().st_size != entry["bytes"]:
                print(f"FAIL {name}: {src.stat().st_size} B {got[:12]}… != pinned "
                      f"{entry['bytes']} B {sha_expect[:12]}…")
                failures += 1
                continue
            print(f"OK   {name}: sha256 matches the pin")
            place(src, out / entry["canonical"], sha_expect, args.verify)

    if failures:
        sys.exit(f"\nFAILED: {failures} verification failure(s) - nothing unsafe was written")

    print("\nBridge verified. Canonical files in", out,
          "next: python scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
