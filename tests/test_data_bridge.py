"""Round-trip test of the git data bridge (scripts/make_data_bridge.py ->
scripts/assemble_data_bridge.py).

The bridge is the only channel that can carry the official competition rasters
into the egress-restricted dev sandbox, so its integrity contract is tested
directly, stdlib-only:

* make:  parts are cut at PART_BYTES, every sha256 is pinned to the inventory,
  and the parts sum to the original byte count;
* assemble: reassembles to bit-identical files (sha256), is idempotent, and
  REFUSES to write the feature stack when any part is corrupted (exit 1).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

NAMES = ["gems-geodawn-numerical-features.tif", "existing_faults.tif",
         "example_submission.tif"]
CANONICAL = {"gems-geodawn-numerical-features.tif": "training_features.tif",
             "existing_faults.tif": "labels.tif",
             "example_submission.tif": "sample_submission.tif"}


def _run(script: str, *args: str, cwd: Path, part_bytes: int | None = None) -> subprocess.CompletedProcess:
    src = (REPO / "scripts" / script).read_text()
    if part_bytes is not None:
        src = src.replace("PART_BYTES = 94_371_840", f"PART_BYTES = {part_bytes}")
    script_path = cwd / script
    script_path.write_text(src)
    return subprocess.run([sys.executable, script_path, *args], cwd=cwd,
                          capture_output=True, text=True)


@pytest.fixture()
def bridge_world(tmp_path: Path):
    """A fake 'runner' workspace: data/<mirror files> + evidence/inventory.json."""
    rng = random.Random(0)
    sizes = {"gems-geodawn-numerical-features.tif": 250_000,
             "existing_faults.tif": 4_096,
             "example_submission.tif": 8_192}
    (tmp_path / "data" / "evidence").mkdir(parents=True)
    files = {}
    for name, n in sizes.items():
        blob = bytes(rng.getrandbits(8) for _ in range(n))
        (tmp_path / "data" / name).write_bytes(blob)
        files[name] = {"bytes": n, "sha256": hashlib.sha256(blob).hexdigest()}
    (tmp_path / "data" / "evidence" / "inventory.json").write_text(
        json.dumps({"generated_utc": "test", "files": [
            {"name": k, **v} for k, v in files.items()]}))
    return tmp_path, files


def test_make_cuts_parts_and_pins_inventory(bridge_world):
    tmp, files = bridge_world
    r = _run("make_data_bridge.py", "--data-dir", "data", "--out", "data/bridge",
             cwd=tmp, part_bytes=100_000)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((tmp / "data" / "bridge" / "manifest.json").read_text())
    by_name = {e["name"]: e for e in manifest["files"]}
    feat = by_name["gems-geodawn-numerical-features.tif"]
    assert len(feat["parts"]) == 3  # 250,000 B at 100,000 B/part
    assert [p["bytes"] for p in feat["parts"]] == [100_000, 100_000, 50_000]
    assert sum(p["bytes"] for p in feat["parts"]) == feat["bytes"]
    for name, meta in files.items():
        assert by_name[name]["sha256"] == meta["sha256"]
        assert by_name[name]["canonical"] == CANONICAL[name]
    # small files are copied whole, not split
    assert "parts" not in by_name["existing_faults.tif"]
    assert (tmp / "data" / "bridge" / "existing_faults.tif").read_bytes() == \
        (tmp / "data" / "existing_faults.tif").read_bytes()


def test_make_refuses_unpinned_bytes(bridge_world):
    tmp, _ = bridge_world
    # silently replace the labels with different bytes -> sha256 no longer pinned
    (tmp / "data" / "existing_faults.tif").write_bytes(os.urandom(4_096))
    r = _run("make_data_bridge.py", "--data-dir", "data", "--out", "data/bridge",
             cwd=tmp, part_bytes=100_000)
    assert r.returncode != 0
    assert "does not match the pinned inventory" in (r.stdout + r.stderr)
    # nothing was written
    assert not (tmp / "data" / "bridge" / "manifest.json").exists()


def _receive(tmp: Path):
    """Simulate the sandbox: only bridge/ + evidence/ exist, run the assembler."""
    recv = tmp / "sandbox"
    shutil.copytree(tmp / "data" / "bridge", recv / "data" / "bridge")
    shutil.copytree(tmp / "data" / "evidence", recv / "data" / "evidence")
    r = _run("assemble_data_bridge.py", cwd=recv)
    return recv, r


def test_assemble_round_trip_bit_identical(bridge_world):
    tmp, files = bridge_world
    assert _run("make_data_bridge.py", "--data-dir", "data", "--out", "data/bridge",
                cwd=tmp, part_bytes=100_000).returncode == 0
    recv, r = _receive(tmp)
    assert r.returncode == 0, r.stderr
    for name, meta in files.items():
        out = recv / "data" / CANONICAL[name]
        assert out.exists()
        assert hashlib.sha256(out.read_bytes()).hexdigest() == meta["sha256"]
    # idempotent second run changes nothing and still passes
    r2 = _run("assemble_data_bridge.py", cwd=recv)
    assert r2.returncode == 0
    for name, meta in files.items():
        assert hashlib.sha256(
            (recv / "data" / CANONICAL[name]).read_bytes()).hexdigest() == meta["sha256"]


def test_assemble_refuses_corrupted_part(bridge_world):
    tmp, files = bridge_world
    assert _run("make_data_bridge.py", "--data-dir", "data", "--out", "data/bridge",
                cwd=tmp, part_bytes=100_000).returncode == 0
    recv, r = _receive(tmp)
    assert r.returncode == 0
    # corrupt one part of the feature stack
    part = recv / "data" / "bridge" / "gems-geodawn-numerical-features.tif.part-001"
    b = bytearray(part.read_bytes())
    b[7] ^= 0xFF
    part.write_bytes(bytes(b))
    # remove the previously assembled output so the write path is exercised
    (recv / "data" / "training_features.tif").unlink()
    r_bad = _run("assemble_data_bridge.py", cwd=recv)
    assert r_bad.returncode != 0
    assert not (recv / "data" / "training_features.tif").exists(), \
        "a corrupted bridge must not produce an output raster"


def test_verify_mode_writes_nothing(bridge_world):
    tmp, _ = bridge_world
    assert _run("make_data_bridge.py", "--data-dir", "data", "--out", "data/bridge",
                cwd=tmp, part_bytes=100_000).returncode == 0
    recv, r = _receive(tmp)
    assert r.returncode == 0
    for name in CANONICAL.values():
        (recv / "data" / name).unlink()
    r_v = _run("assemble_data_bridge.py", "--verify", cwd=recv)
    assert r_v.returncode == 0
    for name in CANONICAL.values():
        assert not (recv / "data" / name).exists()
