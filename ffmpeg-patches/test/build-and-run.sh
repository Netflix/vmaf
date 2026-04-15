#!/usr/bin/env bash
# build-and-run.sh — smoke-test the ffmpeg-patches/ series against a pinned
# upstream FFmpeg SHA.
#
# What it does:
#   1. Fetches FFmpeg at the pinned SHA into $FFMPEG_SRC (clone if absent).
#   2. Applies every patch listed in ffmpeg-patches/series.txt (comment-
#      stripped) in order.
#   3. Configures with the minimum libraries needed for libvmaf + vmaf_pre,
#      builds ffmpeg, and verifies:
#        - `ffmpeg -h filter=libvmaf` lists `tiny_model`
#        - `ffmpeg -h filter=vmaf_pre` exits 0
#   4. Tears down the build tree unless $KEEP_BUILD is set.
#
# Requires libvmaf already installed (`pip`-level: "pkg-config --cflags libvmaf"
# must resolve). Set VMAF_PREFIX to point at a non-standard install prefix.
#
# Patches target FFmpeg n8.1 (the current release at time of authoring).
# Applied via `git apply --3way` so small upstream drift can still merge.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$PATCHES_DIR/.." && pwd)"

: "${FFMPEG_SRC:=/tmp/vmaf-ffmpeg}"
: "${FFMPEG_SHA:=n8.1}"        # pinned release tag; update as patches evolve
: "${KEEP_BUILD:=}"
: "${VMAF_PREFIX:=}"

if [[ -n "$VMAF_PREFIX" ]]; then
    export PKG_CONFIG_PATH="$VMAF_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="$VMAF_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

if ! command -v git >/dev/null; then
    echo "git not found on PATH" >&2; exit 77
fi
if ! pkg-config --exists libvmaf; then
    echo "libvmaf not found via pkg-config — install libvmaf first or set VMAF_PREFIX" >&2
    exit 77
fi

if [[ ! -d "$FFMPEG_SRC/.git" ]]; then
    echo "Cloning FFmpeg into $FFMPEG_SRC …"
    git clone --depth 1 --branch "$FFMPEG_SHA" https://git.ffmpeg.org/ffmpeg.git "$FFMPEG_SRC"
else
    git -C "$FFMPEG_SRC" fetch --tags --depth 1 origin "$FFMPEG_SHA" || true
    git -C "$FFMPEG_SRC" reset --hard FETCH_HEAD 2>/dev/null || \
        git -C "$FFMPEG_SRC" checkout "$FFMPEG_SHA"
    git -C "$FFMPEG_SRC" clean -fdx
fi

echo "Applying patches from $PATCHES_DIR …"
while IFS= read -r line; do
    line="${line%%#*}"
    line="${line// /}"
    [[ -z "$line" ]] && continue
    echo "  → $line"
    git -C "$FFMPEG_SRC" apply --3way "$PATCHES_DIR/$line"
done < "$PATCHES_DIR/series.txt"

echo "Configuring FFmpeg …"
cd "$FFMPEG_SRC"
./configure \
    --disable-doc \
    --disable-debug \
    --disable-programs \
    --enable-ffmpeg \
    --enable-libvmaf \
    --enable-filter=vmaf_pre \
    --enable-gpl

echo "Building FFmpeg …"
make -j"$(nproc)"

echo "Verifying new options …"
./ffmpeg -hide_banner -h filter=libvmaf 2>&1 | grep -q -- 'tiny_model' \
    || { echo "libvmaf filter does not advertise tiny_model"; exit 1; }
./ffmpeg -hide_banner -h filter=vmaf_pre >/dev/null 2>&1 \
    || { echo "vmaf_pre filter not registered"; exit 1; }

echo "PASS: ffmpeg-patches smoke ok"

if [[ -z "$KEEP_BUILD" ]]; then
    echo "Cleaning $FFMPEG_SRC (set KEEP_BUILD=1 to keep)."
    rm -rf "$FFMPEG_SRC"
fi
