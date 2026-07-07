#!/usr/bin/env bash
# Publish the current dev tree to the PUBLIC GitHub mirror (github.com/cebeuq/sprig),
# excluding internal working docs. Snapshot-based: the dev repo keeps its full
# private history + notes; GitHub accumulates a clean history that never contains them.
#
#   bash scripts/push-public.sh "commit message describing the update"
#
# What ships = git-tracked files MINUS the internal paths below, plus the curated
# public landing files in publish/ (README.md, LICENSE).
set -euo pipefail

REPO_URL="https://github.com/cebeuq/sprig"
DEV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MSG="${1:-Sync from dev}"

# Paths kept private (never pushed):
EXCLUDE='^(PROJECT_MEMORY\.md|diagnosis/|release/RELEASE\.md|local_runs/|publish/)'

MIRROR="$(mktemp -d)"
trap 'rm -rf "$MIRROR"' EXIT
git clone -q "$REPO_URL" "$MIRROR"

# Replace the mirror's tracked content with a fresh clean snapshot.
find "$MIRROR" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cd "$DEV"
git ls-files | grep -vE "$EXCLUDE" | while read -r f; do
  mkdir -p "$MIRROR/$(dirname "$f")"
  cp "$f" "$MIRROR/$f"
done
# Curated public landing files (source of truth lives in publish/). The README
# is the exact Hugging Face model card, so its two embedded images must sit at
# the repo-root paths it references (samples.jpg, figures/pipeline.png).
cp "$DEV/publish/README.md" "$MIRROR/README.md"
cp "$DEV/publish/LICENSE"   "$MIRROR/LICENSE"
mkdir -p "$MIRROR/figures"
cp "$DEV/release/model/samples.jpg"          "$MIRROR/samples.jpg"
cp "$DEV/release/model/figures/pipeline.png" "$MIRROR/figures/pipeline.png"

cd "$MIRROR"
git add -A
if git diff --cached --quiet; then
  echo "no changes to publish."
  exit 0
fi
git commit -q -m "$MSG"
git push -q origin main
echo "published -> $REPO_URL  ($MSG)"
