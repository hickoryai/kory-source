#!/usr/bin/env bash
# Smoke-test an image built by scripts/build_reviewer_image.sh.
#
#   scripts/smoke_reviewer_image.sh <image>
#
# 1. Every reviewer command answers --version.
# 2. With no router configuration mounted, the local reviewer does not start
#    and a waiting call stays unclaimed.
# 3. With a router configuration that has no account, the same.
# 4. With a router configuration that has an account, the local reviewer
#    claims one waiting call and answers it. A stand-in router answers the
#    prompt, so no account or model is used.
#
# All configuration here is synthetic. No real login is mounted or read.
set -euo pipefail

image="${1:?usage: $0 <image>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "== reviewer commands answer --version"
docker run --rm --entrypoint python "$image" -m mira.auto_review.reviewer_info --check
docker run --rm --entrypoint router-acp "$image" --version
docker run --rm --entrypoint kory "$image" --version

mkdir -p "$work/smoke" "$work/empty/accounts" "$work/full/accounts/grok@00000000000000000000000000000000"
cp "$root/tests/smoke/local_reviewer_smoke.py" "$root/tests/fixtures/local_reviewer/stub_router.py" "$work/smoke/"
printf '#!/bin/sh\nexec python /smoke/stub_router.py "$@"\n' > "$work/smoke/router-acp"
chmod 0755 "$work/smoke/router-acp"
printf 'router: auto\n' > "$work/empty/router.yaml"
printf 'router: auto\n' > "$work/full/router.yaml"
chmod -R a+rX "$work"

smoke() {
  local expect="$1"
  shift
  docker run --rm --entrypoint python -e KORY_SOURCE_AUTO_REVIEW=1 \
    -v "$work/smoke:/smoke:ro" "$@" "$image" /smoke/local_reviewer_smoke.py --expect "$expect"
}

echo "== no router configuration: the local reviewer stays off"
smoke absent
echo "== router configuration without an account: the local reviewer stays off"
smoke absent -v "$work/empty:/root/.config/router-acp:ro"
echo "== router configuration with an account: the local reviewer claims one call"
smoke claimed -v "$work/full:/root/.config/router-acp:ro" -e ROUTER_ACP_BIN=/smoke/router-acp
echo "reviewer image smoke: ok"
