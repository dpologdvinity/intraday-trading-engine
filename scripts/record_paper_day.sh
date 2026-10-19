#!/usr/bin/env bash
# Evening record of a paper run: commit its daily forward table to this repository and
# back up all paper journals and run state to a private repository.
#
#     scripts/record_paper_day.sh news-picks
#
# HFT_RECORDS points at a clone of the private backup repository
# (default: ~/git/hft-paper-records).
set -euo pipefail
run=${1:-news-picks}
root=$(cd "$(dirname "$0")/.." && pwd)
backup=${HFT_RECORDS:-$HOME/git/hft-paper-records}
cd "$root"

doc="docs/$run-forward.md"
.venv/bin/python scripts/paper_report.py --run "$run" --markdown "$doc" > /dev/null
if ! git ls-files --error-unmatch "$doc" > /dev/null 2>&1 || ! git diff --quiet -- "$doc"; then
  git add -- "$doc"
  git commit -q -m "docs: record the $run paper session of $(date +%F)" -- "$doc"
  git push -q
fi

mkdir -p "$backup/logs" "$backup/state"
rsync -a logs/trade/ "$backup/logs/"
rsync -a .state/trade/ "$backup/state/"
cd "$backup"
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "Paper records $(date +%F)"
  git push -q
fi
