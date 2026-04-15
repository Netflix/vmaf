#!/usr/bin/env bash
#
# scripts/test-matrix.sh
#
# Run `make ci` inside each docker/dev/*.Dockerfile variant. Local analog of
# the GitHub Actions cross-distro matrix.
#
# Usage:
#   bash scripts/test-matrix.sh                 # all images
#   bash scripts/test-matrix.sh ubuntu-24.04    # single image
#   MAKE_TARGET=test-fast bash scripts/test-matrix.sh  # override target

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

MAKE_TARGET="${MAKE_TARGET:-ci}"

declare -a IMAGES=(
    "ubuntu-24.04:docker/dev/ubuntu-24.04.Dockerfile"
    "arch:docker/dev/arch.Dockerfile"
    "fedora-40:docker/dev/fedora-40.Dockerfile"
    "alpine-3.20:docker/dev/alpine-3.20.Dockerfile"
)

# Allow positional filter(s) to restrict which images run.
if [[ $# -gt 0 ]]; then
    declare -a filtered=()
    for want in "$@"; do
        for entry in "${IMAGES[@]}"; do
            tag="${entry%%:*}"
            if [[ "$tag" == "$want" ]]; then
                filtered+=("$entry")
            fi
        done
    done
    IMAGES=("${filtered[@]}")
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found on PATH" >&2
    exit 2
fi

failed=()
for entry in "${IMAGES[@]}"; do
    tag="${entry%%:*}"
    dockerfile="${entry##*:}"
    image="vmaf-dev:$tag"

    echo
    echo "=== build $tag ==="
    docker build -f "$dockerfile" -t "$image" .

    echo
    echo "=== run $tag (make $MAKE_TARGET) ==="
    if ! docker run --rm \
            -v "$REPO_ROOT:/src:rw" -w /src \
            -e MAKE_TARGET="$MAKE_TARGET" \
            "$image" bash -lc "make \"$MAKE_TARGET\""; then
        failed+=("$tag")
    fi
done

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED: ${failed[*]}"
    exit 1
fi
echo "OK: all ${#IMAGES[@]} images green for target \"$MAKE_TARGET\""
