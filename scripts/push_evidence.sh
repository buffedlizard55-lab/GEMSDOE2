#!/usr/bin/env bash
# =============================================================================
# Commit and push staged evidence, surviving the races that evidence jobs create.
#
# WHY THIS EXISTS
#   The evidence jobs land in parallel: block-holdout.yml runs its folds as a
#   matrix, and a single trigger push can fire block-holdout.yml and
#   pseudo-label.yml at the same time (that is what happened on 2026-09-19: runs
#   35451858112 and 35451858126 both started from the same merge commit).  Every
#   one of those jobs rewrites the SAME derived file,
#   data/evidence/block_holdout/fold_gap_summary.json, because each of them has
#   just added a fold that the summary must count.  A plain
#       git commit; git pull --rebase; git push
#   therefore fails the job (and loses the evidence) whenever two of them
#   interleave: the rebase hits a conflict on a file both sides rewrote, and an
#   untracked file the other side added blocks the rebase before it starts.
#
# WHAT IT DOES INSTEAD
#   On a conflict the derived files are REGENERATED from the merged tree rather
#   than resolved from either side.  That is the only correct resolution: the
#   summary is a function of every committed fold*K_generalisation_gap.json, so
#   once another job has landed, neither parent's version is right any more.
#
# USAGE (from a workflow step, after `git add -f ...`)
#   bash scripts/push_evidence.sh "evidence(scope): what landed [skip ci]" [attempts]
#
# Exit codes: 0 pushed (or nothing to push), 1 the push never succeeded.  The
# caller's artifact upload still carries every file, so a 1 loses the commit,
# not the measurement - but it is an error, not a warning, on purpose: committed
# evidence is the only thing a later session can read (runner artifacts expire
# and cannot be downloaded from every environment).
# =============================================================================
set -uo pipefail

MSG="${1:?usage: push_evidence.sh \"<commit message>\" [attempts]}"
ATTEMPTS="${2:-4}"
BRANCH="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"
DERIVED=data/evidence/block_holdout/fold_gap_summary.json
PYTHON="${PYTHON:-python}"

if git diff --cached --quiet; then
  echo "nothing staged - no evidence to push"
  exit 0
fi

# remember exactly what was staged, so a retry re-adds the same files and never
# sweeps up an unrelated large file that happens to sit in the working tree
STAGED="$(git diff --cached --name-only)"
git commit -q -m "$MSG"

# a brand-new branch has nothing to rebase onto
if ! git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  if git push -u origin "HEAD:$BRANCH"; then
    echo "evidence pushed to a new remote branch $BRANCH"
    exit 0
  fi
  echo "::error::could not create the remote branch $BRANCH"
  exit 1
fi

for i in $(seq 1 "$ATTEMPTS"); do
  if git pull --rebase origin "$BRANCH" >/dev/null 2>&1; then
    if git push origin "HEAD:$BRANCH"; then
      echo "evidence pushed (attempt $i of $ATTEMPTS)"
      exit 0
    fi
    echo "::warning::push rejected on attempt $i - another evidence job landed first"
    # our commit is intact; the next iteration rebases it onto the new tip
    sleep 10
    continue
  fi

  echo "::warning::rebase blocked on attempt $i - regenerating the derived files from the merged tree"
  git rebase --abort >/dev/null 2>&1 || true
  # back to the pre-commit state, keeping every new file in the working tree
  git reset --mixed HEAD~1 >/dev/null 2>&1 || true

  # any staged file that HEAD does not have is untracked now, and an untracked
  # file the other job also added blocks the rebase outright - move those aside
  BACKUP="$(mktemp -d)"
  for f in $STAGED; do
    if ! git cat-file -e "HEAD:$f" >/dev/null 2>&1; then
      mkdir -p "$BACKUP/$(dirname "$f")"
      mv "$f" "$BACKUP/$f" >/dev/null 2>&1 || true
    fi
  done
  # the derived summary may be tracked-and-modified: drop our version, the merged
  # tree's version is about to be superseded by a regeneration anyway
  git checkout -- "$DERIVED" >/dev/null 2>&1 || rm -f "$DERIVED"

  if ! git pull --rebase origin "$BRANCH" >/dev/null 2>&1; then
    git rebase --abort >/dev/null 2>&1 || true
    for f in $STAGED; do
      [ -e "$f" ] || { mkdir -p "$(dirname "$f")"; mv "$BACKUP/$f" "$f" >/dev/null 2>&1 || true; }
    done
    rm -rf "$BACKUP"
    sleep 15
    continue
  fi

  # put back the files the merged tree did not bring (our own per-fold evidence)
  for f in $STAGED; do
    if [ ! -e "$f" ] && [ -e "$BACKUP/$f" ]; then
      mkdir -p "$(dirname "$f")"
      mv "$BACKUP/$f" "$f" >/dev/null 2>&1 || true
    fi
  done
  rm -rf "$BACKUP"

  # re-derive the shared summary now that every landed fold is in this tree
  if "$PYTHON" scripts/read_landed_reports.py --gaps-only --quiet >/dev/null 2>&1; then
    echo "regenerated $DERIVED from the merged tree"
  else
    echo "::warning::$DERIVED could not be regenerated - keeping whichever version the merged tree has"
  fi

  for f in $STAGED; do git add -f "$f" >/dev/null 2>&1 || true; done
  if git diff --cached --quiet; then
    echo "nothing left to commit after regeneration - the other job committed the same evidence"
    exit 0
  fi
  git commit -q -m "$MSG"
  sleep 5
done

echo "::error::evidence not pushed after $ATTEMPTS attempts ($MSG) - the run artifact still carries every file"
exit 1
