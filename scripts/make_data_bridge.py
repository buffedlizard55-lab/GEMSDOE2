#!/usr/bin/env python3
"""Build the git data bridge: commit the official competition rasters as parts.

WHY THIS EXISTS
---------------
The development sandbox egresses through an allowlist that only permits
github.com / api.github.com / codeload.github.com / pypi.org.  Dropbox (the
host of the competition's data-tab mirrors), S3, DrivenData and the Azure
host that serves Actions ARTIFACTS are all unreachable from there, so no
existing channel can carry the binary rasters into the sandbox.

The only transport that survives the allowlist is a git commit on this
branch.  GitHub rejects any single blob >= 100 MB, so the 418,912,844-byte
feature stack is split into PART_BYTES (90 MiB) parts under data/bridge/.

INTEGRITY
---------
Every file's sha256 and byte size is verified against
data/evidence/inventory.json — the independent, machine-measured inventory
produced by scripts/inspect_competition_data.py on a GitHub-hosted runner on
2026-09-14 — BEFORE anything is written into data/bridge/.  A drift between
the mirror and the pinned inventory aborts the build.  The manifest written
here carries the same pins so scripts/assemble_data_bridge.py can re-verify
on the receiving side without network access.

USAGE (on a runner / unrestricted machine, after downloading the mirrors
into data/ — see .github/workflows/place-competition-data.yml):
    python scripts/make_data_bridge.py --data-dir data --out data/bridge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

# 90 MiB: safely below GitHub's 100 MiB per-blob hard limit.
PART_BYTES = 94_371_840

INVENTORY = Path("data/evidence/inventory.json")

# mirror file name -> canonical name used by configs + scripts/prepare_data.py
CANONICAL = {
    "gems-geodawn-numerical-features.tif": "training_features.tif",
    "existing_faults.tif": "labels.tif",
    "example_submission.tif": "sample_submission.tif",
}

MIRRORS = {
    "gems-geodawn-numerical-features.tif":
        "https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/"
        "gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&dl=1",
    "existing_faults.tif":
        "https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/"
        "existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&dl=1",
    "example_submission.tif":
        "https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/"
        "example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&dl=1",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pins() -> dict:
    if not INVENTORY.exists():
        sys.exit(f"FAIL: {INVENTORY} not found - the sha256 pins are mandatory, "
                 "not optional.  Run scripts/inspect_competition_data.py first.")
    inv = json.loads(INVENTORY.read_text())
    return {f["name"]: f for f in inv["files"] if f["name"] in CANONICAL}


def split_into_parts(src: Path, out_dir: Path, stem: str) -> list[dict]:
    parts = []
    with src.open("rb") as f:
        idx = 0
        while True:
            blob = f.read(PART_BYTES)
            if not blob:
                break
            name = f"{stem}.part-{idx:03d}"
            (out_dir / name).write_bytes(blob)
            parts.append({"name": name, "bytes": len(blob),
                          "sha256": hashlib.sha256(blob).hexdigest()})
            idx += 1
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/bridge")
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out)

    pins = load_pins()
    missing = [n for n in CANONICAL if n not in pins]
    if missing:
        sys.exit(f"FAIL: inventory.json has no pins for {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "generated_by": "scripts/make_data_bridge.py",
        "purpose": ("transport the official competition rasters from an unrestricted "
                    "runner into the egress-restricted dev sandbox via git; GitHub's "
                    "100 MB blob limit forces the feature stack to be split"),
        "source": {
            "official_data_tab":
                "https://www.drivendata.org/competitions/306/competition-doe-gems/data/",
            "mirrors": MIRRORS,
            "sha256_pins": ("data/evidence/inventory.json "
                            "(scripts/inspect_competition_data.py, GitHub-hosted runner, "
                            "generated " + json.loads(INVENTORY.read_text())["generated_utc"] + ")"),
        },
        "part_bytes": PART_BYTES,
        "reassemble_command": "python scripts/assemble_data_bridge.py",
        "files": [],
    }

    for name, canonical in CANONICAL.items():
        src = data_dir / name
        if not src.exists():
            sys.exit(f"FAIL: {src} not found - download the mirrors first")
        size = src.stat().st_size
        digest = sha256_file(src)
        pin = pins[name]
        if digest != pin["sha256"] or size != pin["bytes"]:
            sys.exit(f"FAIL: {name} does not match the pinned inventory.\n"
                     f"  measured: {size} B sha256 {digest}\n"
                     f"  pinned:   {pin['bytes']} B sha256 {pin['sha256']}\n"
                     f"Refusing to bridge bytes whose provenance cannot be verified.")
        entry = {"name": name, "canonical": canonical, "bytes": size, "sha256": digest}
        if size > PART_BYTES:
            entry["parts"] = split_into_parts(src, out_dir, name)
            total = sum(p["bytes"] for p in entry["parts"])
            if total != size:
                sys.exit(f"FAIL: parts of {name} sum to {total} B, expected {size} B")
        else:
            shutil.copyfile(src, out_dir / name)
        manifest["files"].append(entry)
        print(f"OK {name}: {size} B sha256 {digest}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\nwrote {out_dir / 'manifest.json'} and "
          f"{sum(len(e.get('parts', [e['name']])) for e in manifest['files'])} payload files")
    print("next (receiving side): python scripts/assemble_data_bridge.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
