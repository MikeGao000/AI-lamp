#!/usr/bin/env bash
# Safe Pi deployment: fast-forward to the reviewed repository version, then
# run the full software regression suite. It never starts real motor hardware.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

component="${1:-}"
case "$component" in
  reading|safety|motion|can|autonomous|system) ;;
  *)
    echo "Usage: bash scripts/update_and_test.sh {reading|safety|motion|can|autonomous|system} [runner options]"
    echo "For a release-only full scan: python3 run_full_test_suite.py --profile full"
    exit 2
    ;;
esac
shift

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "No Git remote named 'origin' is configured. Connect this project to its private repository first."
  exit 2
fi

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  echo "Repository is not on a named branch; refusing to update."
  exit 2
fi

git pull --ff-only origin "$branch"
# Code always updates as one Git version; tests run only for the named changed
# subsystem. Full scanning remains an explicit release/safety action.
exec python3 run_full_test_suite.py --component "$component" "$@"
