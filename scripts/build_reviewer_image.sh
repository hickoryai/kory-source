#!/usr/bin/env bash
# Build the image with the bundled reviewer at the newest version of everything.
#
#   scripts/build_reviewer_image.sh <tag> [<tag> ...]
#
# Resolves the newest router and agent CLI commits, agent package versions and
# base image digests (scripts/resolve_reviewer_deps.py), checks both source
# trees out at those commits, and builds the image from them. The image is
# loaded into the local daemon and never pushed. Nothing here is cached
# between releases except Docker's own build cache.
#
# Environment:
#   REVIEWER_MANIFEST   write the resolved manifest here (default: reviewer-manifest.json)
#   REVIEWER_OVERRIDES  space-separated NAME=VALUE pins passed to the resolver
#   IMAGE_LABELS        newline-separated extra labels
#   BUILDX_ARGS         extra arguments for `docker buildx build` (cache settings)
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <tag> [<tag> ...]" >&2
  exit 2
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
manifest="${REVIEWER_MANIFEST:-reviewer-manifest.json}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

overrides=()
for pin in ${REVIEWER_OVERRIDES:-}; do
  overrides+=(--override "$pin")
done
python3 "$root/scripts/resolve_reviewer_deps.py" --dockerfile "$root/Dockerfile" \
  --output "$manifest" "${overrides[@]}"
python3 "$root/scripts/resolve_reviewer_deps.py" --markdown "$manifest"

# One shallow fetch of exactly the resolved commit. The git metadata is
# removed so the build context is the source tree alone.
checkout() {
  local repo="$1" revision="$2" dest="$3"
  git init --quiet "$dest"
  git -C "$dest" fetch --quiet --depth 1 "$repo" "$revision"
  git -C "$dest" checkout --quiet --detach FETCH_HEAD
  test "$(git -C "$dest" rev-parse HEAD)" = "$revision"
  rm -rf "$dest/.git"
}
read_manifest() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$manifest" "$1"; }
checkout "${ROUTER_ACP_REPO:-https://github.com/zaneclaes/router-acp}" "$(read_manifest router_acp)" "$work/router-acp"
checkout "${AGENT_CLI_REPO:-https://github.com/hickoryai/kory-cli}" "$(read_manifest agent_cli)" "$work/agent-cli"

args=(--load --provenance=false --sbom=false
  --build-context "router-acp-src=$work/router-acp"
  --build-context "agent-cli-src=$work/agent-cli")
while IFS= read -r build_arg; do
  args+=(--build-arg "$build_arg")
done < <(python3 "$root/scripts/resolve_reviewer_deps.py" --build-args "$manifest")
for tag in "$@"; do
  args+=(--tag "$tag")
done
while IFS= read -r label; do
  if [ -n "$label" ]; then args+=(--label "$label"); fi
done <<< "${IMAGE_LABELS:-}"
# shellcheck disable=SC2086
docker buildx build "${args[@]}" ${BUILDX_ARGS:-} "$root"
