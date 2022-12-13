#!/bin/sh -x
set -e

# no gpumask: use cuda
./tools/vmaf \
    --reference /dev/zero \
    --distorted /dev/zero \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --frame_cnt 2 \
    --gpumask 0

# gpumask: use cpu
./tools/vmaf \
    --reference /dev/zero \
    --distorted /dev/zero \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --frame_cnt 2 \
    --gpumask -1

# no gpumask: use cuda for vmaf features, cpu for psnr
./tools/vmaf \
    --reference /dev/zero \
    --distorted /dev/zero \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --frame_cnt 2 \
    --gpumask 0 \
    --feature psnr \
    --output /dev/stdout

# gpumask: use cpu for vmaf features and psnr
./tools/vmaf \
    --reference /dev/zero \
    --distorted /dev/zero \
    --width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
    --frame_cnt 2 \
    --gpumask -1 \
    --feature psnr
