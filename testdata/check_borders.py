#!/usr/bin/env python3
"""Compute DWT subband dimensions and ADM borders for 576x324 at each scale,
to verify SYCL vs CPU iteration domain."""
import math

w, h = 576, 324
ADM_BORDER_FACTOR = 0.2

for scale in range(4):
    # DWT halves dimensions each level (ceil for odd)
    cur_w = w
    cur_h = h
    for i in range(scale + 1):
        cur_w = (cur_w + 1) // 2
        cur_h = (cur_h + 1) // 2

    # CSF den borders (used for both CSF den and CM in SYCL)
    left = int(cur_w * ADM_BORDER_FACTOR - 0.5)
    top = int(cur_h * ADM_BORDER_FACTOR - 0.5)
    right = cur_w - left
    bottom = cur_h - top
    active_w = right - left
    active_h = bottom - top

    # CM borders (CPU uses start_col/end_col)
    start_col = max(left, 1)
    end_col = min(right, cur_w - 1)
    start_row = max(top, 1)
    end_row = min(bottom, cur_h - 1)
    cm_w = end_col - start_col
    cm_h = end_row - start_row

    print(f"Scale {scale}: band={cur_w}x{cur_h}")
    print(f"  CSF den domain: left={left} top={top} right={right} bottom={bottom} active={active_w}x{active_h}")
    print(f"  CM domain (CPU): start_col={start_col} end_col={end_col} start_row={start_row} end_row={end_row}")
    print(f"    CM pixels: {cm_w}x{cm_h} = {cm_w * cm_h}")
    print(f"  SYCL CM iterates: col=[{left}..{right}) row=[{top}..{bottom})")
    print(f"    SYCL CM pixels: {active_w}x{active_h} = {active_w * active_h}")
    if cm_w != active_w or cm_h != active_h:
        print(f"    *** MISMATCH: CPU CM domain != SYCL CM domain! ***")
        print(f"    CPU: {cm_w}x{cm_h} = {cm_w*cm_h}, SYCL: {active_w}x{active_h} = {active_w*active_h}")
    print()
