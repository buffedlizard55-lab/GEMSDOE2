"""Shared helper for workflow-structure tests: look only at lines that actually execute.

Why: a first pass of these assertions scanned whole `run:` blocks and so flagged the `|` inside a
markdown table printed by `echo`, the string "torch" inside a comment explaining why torch is not
installed, and the `|| true` of a disk-cleanup step that is allowed to fail. Those are not defects,
and a test that cries wolf is worse than no test.
"""
from __future__ import annotations

import re


def executable_lines(run_body: str) -> list[str]:
    out = []
    for raw in (run_body or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("echo ", "printf ", 'echo"')) or ">> \"$GITHUB_STEP_SUMMARY\"" in line:
            continue                      # prose the job prints, not a command that can fail
        if re.match(r"^\|", line) or line.startswith('```'):
            continue
        out.append(line)
    return out


def pipes(line: str) -> bool:
    """A real pipe: not `||`."""
    return bool(re.search(r"(?<!\|)\|(?!\|)", line))
