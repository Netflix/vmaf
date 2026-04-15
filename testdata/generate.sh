#!/bin/bash
# Generate large-resolution test YUV pairs from Big Buck Bunny source.
# Small resolutions (576x324, 640x480) are committed to the repo.
# This script generates 720p, 1080p, and 4K pairs that are gitignored.
#
# Usage: ./generate.sh [path/to/bbb.mp4]
#
# If no path given, looks for bbb_sunflower_2160p_30fps_normal.mp4 in
# ../.workingdir/ or downloads it.

set -euo pipefail
cd "$(dirname "$0")"

FRAMES=48
BBB="${1:-../.workingdir/bbb_sunflower_2160p_30fps_normal.mp4}"

if [ ! -f "$BBB" ]; then
    echo "ERROR: Source video not found at: $BBB"
    echo "Download Big Buck Bunny 4K from:"
    echo "  https://download.blender.org/demo/movies/BBB/bbb_sunflower_2160p_30fps_normal.mp4"
    echo "Or pass the path as an argument: $0 /path/to/bbb.mp4"
    exit 1
fi

RESOLUTIONS="1280x720 1920x1080 3840x2160"

for RES in $RESOLUTIONS; do
    W="${RES%x*}"
    H="${RES#*x}"
    REF="ref_${W}x${H}_${FRAMES}f.yuv"
    DIS="dis_${W}x${H}_${FRAMES}f.yuv"

    if [ -f "$REF" ] && [ -f "$DIS" ]; then
        echo "SKIP $RES — already exists"
        continue
    fi

    echo "Generating $RES ref (lossless decode)..."
    ffmpeg -y -loglevel warning -i "$BBB" \
        -vf "scale=${W}:${H}" -pix_fmt yuv420p \
        -frames:v "$FRAMES" "$REF"

    echo "Generating $RES dis (CRF 28 encode → decode)..."
    ffmpeg -y -loglevel warning -i "$BBB" \
        -vf "scale=${W}:${H}" -pix_fmt yuv420p \
        -c:v libx264 -crf 28 -preset fast -an \
        -frames:v "$FRAMES" -f mp4 - | \
    ffmpeg -y -loglevel warning -i - \
        -pix_fmt yuv420p -frames:v "$FRAMES" "$DIS"

    echo "  $REF: $(du -h "$REF" | cut -f1)"
    echo "  $DIS: $(du -h "$DIS" | cut -f1)"
done

echo "Done. All test data ready in $(pwd)/"
