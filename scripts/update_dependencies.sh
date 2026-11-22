#!/usr/bin/env bash
# Update this fork to the latest of everything, on a branch, as a draft PR.
#
#   scripts/update_dependencies.sh [--continue] [--image-in-ci] [--no-pr]
#
# Run from a clean checkout of an up-to-date `main`. In order:
#
#   1. Start the branch `update-dependencies/<date>`.
#   2. Merge upstream `main` with a merge commit. Upstream is ALWAYS merged.
#      This script never rebases, never force-pushes, and never writes to
#      `main`. If the merge conflicts it stops and lists the files: resolve
#      them, commit the merge, and run again with --continue.
#   3. Upgrade the Python lockfile (`uv lock --upgrade`) and the UI lockfile
#      (`npm update` and `npm audit fix`, inside the ranges in package.json).
#   4. Resolve the newest bundled reviewer versions, for the record only:
#      every release resolves them again when it builds.
#   5. Record the new upstream baseline in FORK.md and the refresh in
#      CHANGELOG.md.
#   6. Run lint, the full test suite, the UI build, the source audit, and the
#      image build, layer audit and reviewer smoke test.
#   7. Commit, push the branch, and open a draft PR.
#
# It stops at the first failure and opens no PR. It never tags and never
# publishes: see "Update to the latest dependencies" in AGENTS.md for the
# steps after the PR.
#
#   --continue     resume after you resolved and committed a conflicted merge
#   --image-in-ci  skip the local image build; the PR's "Reviewer image" check
#                  builds, audits and smoke-tests it instead
#   --no-pr        stop after the local commit
#
# Environment: UPSTREAM_REPO (default https://github.com/miracodeai/mira),
# UPSTREAM_BRANCH (default main).
set -euo pipefail

upstream_repo="${UPSTREAM_REPO:-https://github.com/miracodeai/mira}"
upstream_branch="${UPSTREAM_BRANCH:-main}"
resume=0
image_in_ci=0
open_pr=1
for arg in "$@"; do
  case "$arg" in
    --continue) resume=1 ;;
    --image-in-ci) image_in_ci=1 ;;
    --no-pr) open_pr=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
today="$(date -u +%Y-%m-%d)"
state_dir="$(git rev-parse --git-dir)/update-dependencies"

step() { printf '\n== %s\n' "$*"; }
stop() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }

if [ -n "$(git status --porcelain)" ]; then
  stop "the working tree is not clean. Commit or discard changes first."
fi

if [ "$resume" = "0" ]; then
  step "Check main is current"
  [ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || stop "start from the main branch."
  git fetch --quiet origin main
  [ "$(git rev-parse HEAD)" = "$(git rev-parse FETCH_HEAD)" ] \
    || stop "local main is not origin/main. Run: git pull --ff-only origin main"

  branch="update-dependencies/$today"
  step "Start branch $branch"
  git switch --quiet -c "$branch"

  step "Merge upstream $upstream_branch with a merge commit"
  git fetch --quiet "$upstream_repo" "$upstream_branch"
  upstream_sha="$(git rev-parse FETCH_HEAD)"
  mkdir -p "$state_dir"
  printf '%s\n' "$upstream_sha" > "$state_dir/upstream"
  if git merge-base --is-ancestor "$upstream_sha" HEAD; then
    echo "upstream $upstream_sha is already merged"
  elif ! git merge --no-ff --no-edit -m "Merge upstream $upstream_branch at ${upstream_sha:0:12}" "$upstream_sha"; then
    echo
    echo "The merge conflicts in:"
    git diff --name-only --diff-filter=U | sed 's/^/  /'
    echo
    echo "Resolve them inside the fork-delta policy in FORK.md, then:"
    echo "  git add -A && git commit --no-edit"
    echo "  scripts/update_dependencies.sh --continue"
    echo "Do not rebase and do not abandon the merge commit."
    exit 3
  fi
else
  step "Resume after a resolved merge"
  case "$(git rev-parse --abbrev-ref HEAD)" in
    update-dependencies/*) ;;
    *) stop "--continue runs on the update-dependencies/<date> branch." ;;
  esac
  [ -f "$state_dir/upstream" ] || stop "no update is in progress."
  [ ! -f "$(git rev-parse --git-dir)/MERGE_HEAD" ] || stop "the merge is not committed yet."
  upstream_sha="$(cat "$state_dir/upstream")"
  git merge-base --is-ancestor "$upstream_sha" HEAD \
    || stop "upstream $upstream_sha is not merged into this branch."
fi
branch="$(git rev-parse --abbrev-ref HEAD)"

step "Upgrade Python dependencies"
uv lock --upgrade
uv sync --extra dev --extra serve --extra bedrock

step "Upgrade UI dependencies"
npm --prefix ui/mira update --no-audit --no-fund
npm --prefix ui/mira audit fix --no-fund || echo "npm audit fix left findings; review them in the PR"
npm --prefix ui/mira ci --no-audit --no-fund

step "Resolve the newest bundled reviewer versions"
manifest="$(mktemp)"
python3 scripts/resolve_reviewer_deps.py --output "$manifest"
table="$(python3 scripts/resolve_reviewer_deps.py --markdown "$manifest")"
echo "$table"

step "Record the upstream baseline and the refresh"
python3 - "$upstream_sha" "$today" "$table" <<'PY'
import re
import sys
from pathlib import Path

upstream, today, table = sys.argv[1:4]
fork = Path("FORK.md")
text = fork.read_text(encoding="utf-8")
text, count = re.subn(
    r"^- upstream baseline: `[0-9a-f]{40}`;$",
    f"- upstream baseline: `{upstream}`;",
    text,
    count=1,
    flags=re.M,
)
if count != 1:
    sys.exit("FORK.md has no upstream baseline line to update")
marker = "<!-- upstream-sync-log -->"
if marker not in text:
    sys.exit("FORK.md has no upstream sync log")
entry = f"| {today} | `{upstream}` |"
if entry not in text:
    text = text.replace(marker, f"{entry}\n{marker}", 1)
fork.write_text(text, encoding="utf-8")

changelog = Path("CHANGELOG.md")
text = changelog.read_text(encoding="utf-8")
heading = "## [Unreleased]\n"
if heading not in text:
    sys.exit("CHANGELOG.md has no [Unreleased] section")
indented = "\n".join("  " + line for line in table.strip().splitlines())
note = (
    f"\n### Dependencies\n\n"
    f"- **Dependencies refreshed on {today}.** Upstream `main` is merged at `{upstream}`. "
    f"The Python and UI lockfiles are upgraded. The bundled reviewer versions at refresh "
    f"time are below. Each release resolves them again when it builds, and its release "
    f"notes list what it shipped.\n\n{indented}\n"
)
changelog.write_text(text.replace(heading, heading + note, 1), encoding="utf-8")
PY

step "Lint"
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts

step "Full test suite"
uv run pytest

step "UI build"
npm --prefix ui/mira run build

step "Audit build inputs"
python3 scripts/audit_public_image.py --source .

if [ "$image_in_ci" = "1" ]; then
  echo "Image build skipped here. The PR's Reviewer image check must pass before merge."
else
  command -v docker > /dev/null || stop "docker is not available. Re-run with --image-in-ci."
  step "Build, audit and smoke-test the image"
  image="kory-source:update-$today"
  archive="$(mktemp)"
  REVIEWER_MANIFEST="$manifest" scripts/build_reviewer_image.sh "$image"
  docker save "$image" -o "$archive"
  python3 scripts/audit_public_image.py --image "$archive"
  rm -f "$archive"
  scripts/smoke_reviewer_image.sh "$image"
fi

step "Commit"
git add -A
if git diff --cached --quiet; then
  echo "nothing changed beyond the merge"
else
  git commit --quiet -m "Refresh dependencies ($today)"
fi
rm -rf "$state_dir"

drift="$(git diff --stat "$upstream_sha" -- src/ tests/ ui/ | tail -n 1)"
body="$(mktemp)"
{
  echo "Refreshes this fork to the latest dependencies."
  echo
  echo "- Upstream \`$upstream_branch\` merged at \`$upstream_sha\` with a merge commit."
  echo "- Python lockfile upgraded with \`uv lock --upgrade\`; UI lockfile with \`npm update\`."
  echo "- Lint, the full test suite, the UI build and the source audit passed locally."
  if [ "$image_in_ci" = "1" ]; then
    echo "- The image was NOT built locally. The Reviewer image check must pass before merge."
  else
    echo "- The image build, layer audit and reviewer smoke test passed locally."
  fi
  echo "- Drift from upstream (\`git diff --stat $upstream_sha -- src/ tests/ ui/\`): ${drift:-none}"
  echo
  echo "**Merge this PR with a merge commit, not a squash or a rebase.** Upstream's commits"
  echo "must stay ancestors of \`main\`, or the next upstream merge starts from the wrong base."
  echo
  echo "Bundled reviewer versions at refresh time (each release resolves them again):"
  echo
  echo "$table"
} > "$body"

if [ "$open_pr" = "0" ]; then
  echo
  echo "Committed on $branch. PR body: $body"
  exit 0
fi

step "Push the branch and open a draft PR"
git push --set-upstream origin "$branch"
gh pr create --draft --base main --head "$branch" \
  --title "Refresh dependencies ($today)" --body-file "$body"
echo
echo "Draft PR opened. Next steps are in AGENTS.md, \"Update to the latest dependencies\"."
