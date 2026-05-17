/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal compute kernel for integer_moment (T8-1k / ADR-0421).
 *  Emits four per-frame statistics across ref and dis planes:
 *  float_moment_ref1st, float_moment_dis1st,
 *  float_moment_ref2nd, float_moment_dis2nd.
 *
 *  Algorithm (must match CPU libvmaf/src/feature/float_moment.c and
 *  all other GPU backends — CUDA/SYCL/Vulkan/HIP):
 *    For each plane in {ref, dis}:
 *      1st moment: sum  += val              (raw integer pixel)
 *      2nd moment: sum2 += val * val        (raw integer squared)
 *    Host: divide each accumulated sum by (W * H).
 *
 *  MSL lacks atomic_ulong, so each WG emits lo/hi uint32 halves for
 *  each of the four accumulators (8 uint32 values per WG total).
 *  Buffer layout per WG at index idx = bid.y * grid_groups.x + bid.x:
 *    sums_lo[idx * 4 + 0] = (uint32)(ref1 & 0xFFFFFFFF)
 *    sums_hi[idx * 4 + 0] = (uint32)(ref1 >> 32)
 *    sums_lo[idx * 4 + 1] = (uint32)(dis1 & 0xFFFFFFFF)
 *    sums_hi[idx * 4 + 1] = (uint32)(dis1 >> 32)
 *    sums_lo[idx * 4 + 2] = (uint32)(ref2 & 0xFFFFFFFF)
 *    sums_hi[idx * 4 + 2] = (uint32)(ref2 >> 32)
 *    sums_lo[idx * 4 + 3] = (uint32)(dis2 & 0xFFFFFFFF)
 *    sums_hi[idx * 4 + 3] = (uint32)(dis2 >> 32)
 *
 *  Host reconstructs: val = ((uint64)hi << 32) | lo, accumulates in
 *  double (53-bit mantissa — sufficient for 64-bit integers up to ~9e15).
 *
 *  Buffer bindings:
 *   [[buffer(0)]] ref      — const uchar * (byte-addressed)
 *   [[buffer(1)]] dis      — const uchar *
 *   [[buffer(2)]] sums_lo  — uint * (grid_w × grid_h × 4 uint32 lo halves)
 *   [[buffer(3)]] sums_hi  — uint * (grid_w × grid_h × 4 uint32 hi halves)
 *   [[buffer(4)]] strides  — uint2 (8bpc: .x=ref_stride, .y=dis_stride)
 *                           — uint4 (16bpc: .x=ref, .y=dis, .z=bpc, .w=0)
 *   [[buffer(5)]] dim      — uint2 (width, height)
 */

#include <metal_stdlib>
using namespace metal;

/* ------------------------------------------------------------------ */
/*  8 bpc kernel                                                        */
/* ------------------------------------------------------------------ */
kernel void integer_moment_kernel_8bpc(
    const device uchar  *ref      [[buffer(0)]],
    const device uchar  *dis      [[buffer(1)]],
    device       uint   *sums_lo  [[buffer(2)]],
    device       uint   *sums_hi  [[buffer(3)]],
    constant     uint2  &strides  [[buffer(4)]],
    constant     uint2  &dim      [[buffer(5)]],
    uint2  gid         [[thread_position_in_grid]],
    uint2  bid         [[threadgroup_position_in_grid]],
    uint2  grid_groups [[threadgroups_per_grid]],
    uint   lid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_id     [[simdgroup_index_in_threadgroup]],
    uint   simd_count  [[simdgroups_per_threadgroup]])
{
    const int width  = (int)dim.x;
    const int height = (int)dim.y;

    ulong my_r1 = 0uL, my_d1 = 0uL, my_r2 = 0uL, my_d2 = 0uL;
    if ((int)gid.x < width && (int)gid.y < height) {
        const ulong rv = (ulong)ref[(int)gid.y * (int)strides.x + (int)gid.x];
        const ulong dv = (ulong)dis[(int)gid.y * (int)strides.y + (int)gid.x];
        my_r1 = rv;
        my_d1 = dv;
        my_r2 = rv * rv;
        my_d2 = dv * dv;
    }

    /* Two-level reduction with uint32 hi/lo split (mirrors integer_psnr.metal). */
    threadgroup uint sg_r1_lo[8], sg_r1_hi[8];
    threadgroup uint sg_d1_lo[8], sg_d1_hi[8];
    threadgroup uint sg_r2_lo[8], sg_r2_hi[8];
    threadgroup uint sg_d2_lo[8], sg_d2_hi[8];

    const uint lane_r1_lo = simd_sum((uint)(my_r1 & 0xFFFFFFFFuL));
    const uint lane_r1_hi = simd_sum((uint)(my_r1 >> 32uL));
    const uint lane_d1_lo = simd_sum((uint)(my_d1 & 0xFFFFFFFFuL));
    const uint lane_d1_hi = simd_sum((uint)(my_d1 >> 32uL));
    const uint lane_r2_lo = simd_sum((uint)(my_r2 & 0xFFFFFFFFuL));
    const uint lane_r2_hi = simd_sum((uint)(my_r2 >> 32uL));
    const uint lane_d2_lo = simd_sum((uint)(my_d2 & 0xFFFFFFFFuL));
    const uint lane_d2_hi = simd_sum((uint)(my_d2 >> 32uL));

    if (simd_lane == 0) {
        sg_r1_lo[simd_id] = lane_r1_lo;
        sg_r1_hi[simd_id] = lane_r1_hi;
        sg_d1_lo[simd_id] = lane_d1_lo;
        sg_d1_hi[simd_id] = lane_d1_hi;
        sg_r2_lo[simd_id] = lane_r2_lo;
        sg_r2_hi[simd_id] = lane_r2_hi;
        sg_d2_lo[simd_id] = lane_d2_lo;
        sg_d2_hi[simd_id] = lane_d2_hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        ulong wg_r1 = 0uL, wg_d1 = 0uL, wg_r2 = 0uL, wg_d2 = 0uL;
        for (uint i = 0; i < simd_count; ++i) {
            wg_r1 += ((ulong)sg_r1_hi[i] << 32uL) | (ulong)sg_r1_lo[i];
            wg_d1 += ((ulong)sg_d1_hi[i] << 32uL) | (ulong)sg_d1_lo[i];
            wg_r2 += ((ulong)sg_r2_hi[i] << 32uL) | (ulong)sg_r2_lo[i];
            wg_d2 += ((ulong)sg_d2_hi[i] << 32uL) | (ulong)sg_d2_lo[i];
        }
        const uint base = (bid.y * grid_groups.x + bid.x) * 4u;
        sums_lo[base + 0] = (uint)(wg_r1 & 0xFFFFFFFFuL);
        sums_hi[base + 0] = (uint)(wg_r1 >> 32uL);
        sums_lo[base + 1] = (uint)(wg_d1 & 0xFFFFFFFFuL);
        sums_hi[base + 1] = (uint)(wg_d1 >> 32uL);
        sums_lo[base + 2] = (uint)(wg_r2 & 0xFFFFFFFFuL);
        sums_hi[base + 2] = (uint)(wg_r2 >> 32uL);
        sums_lo[base + 3] = (uint)(wg_d2 & 0xFFFFFFFFuL);
        sums_hi[base + 3] = (uint)(wg_d2 >> 32uL);
    }
}

/* ------------------------------------------------------------------ */
/*  16 bpc kernel                                                       */
/* ------------------------------------------------------------------ */
kernel void integer_moment_kernel_16bpc(
    const device uchar  *ref      [[buffer(0)]],
    const device uchar  *dis      [[buffer(1)]],
    device       uint   *sums_lo  [[buffer(2)]],
    device       uint   *sums_hi  [[buffer(3)]],
    constant     uint4  &strides  [[buffer(4)]],
    constant     uint2  &dim      [[buffer(5)]],
    uint2  gid         [[thread_position_in_grid]],
    uint2  bid         [[threadgroup_position_in_grid]],
    uint2  grid_groups [[threadgroups_per_grid]],
    uint   lid         [[thread_index_in_threadgroup]],
    uint   simd_lane   [[thread_index_in_simdgroup]],
    uint   simd_id     [[simdgroup_index_in_threadgroup]],
    uint   simd_count  [[simdgroups_per_threadgroup]])
{
    const int width  = (int)dim.x;
    const int height = (int)dim.y;

    ulong my_r1 = 0uL, my_d1 = 0uL, my_r2 = 0uL, my_d2 = 0uL;
    if ((int)gid.x < width && (int)gid.y < height) {
        const device ushort *ref_row =
            (const device ushort *)(ref + (int)gid.y * (int)strides.x);
        const device ushort *dis_row =
            (const device ushort *)(dis + (int)gid.y * (int)strides.y);
        const ulong rv = (ulong)ref_row[(int)gid.x];
        const ulong dv = (ulong)dis_row[(int)gid.x];
        my_r1 = rv;
        my_d1 = dv;
        my_r2 = rv * rv;
        my_d2 = dv * dv;
    }

    threadgroup uint sg_r1_lo[8], sg_r1_hi[8];
    threadgroup uint sg_d1_lo[8], sg_d1_hi[8];
    threadgroup uint sg_r2_lo[8], sg_r2_hi[8];
    threadgroup uint sg_d2_lo[8], sg_d2_hi[8];

    const uint lane_r1_lo = simd_sum((uint)(my_r1 & 0xFFFFFFFFuL));
    const uint lane_r1_hi = simd_sum((uint)(my_r1 >> 32uL));
    const uint lane_d1_lo = simd_sum((uint)(my_d1 & 0xFFFFFFFFuL));
    const uint lane_d1_hi = simd_sum((uint)(my_d1 >> 32uL));
    const uint lane_r2_lo = simd_sum((uint)(my_r2 & 0xFFFFFFFFuL));
    const uint lane_r2_hi = simd_sum((uint)(my_r2 >> 32uL));
    const uint lane_d2_lo = simd_sum((uint)(my_d2 & 0xFFFFFFFFuL));
    const uint lane_d2_hi = simd_sum((uint)(my_d2 >> 32uL));

    if (simd_lane == 0) {
        sg_r1_lo[simd_id] = lane_r1_lo;
        sg_r1_hi[simd_id] = lane_r1_hi;
        sg_d1_lo[simd_id] = lane_d1_lo;
        sg_d1_hi[simd_id] = lane_d1_hi;
        sg_r2_lo[simd_id] = lane_r2_lo;
        sg_r2_hi[simd_id] = lane_r2_hi;
        sg_d2_lo[simd_id] = lane_d2_lo;
        sg_d2_hi[simd_id] = lane_d2_hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
        ulong wg_r1 = 0uL, wg_d1 = 0uL, wg_r2 = 0uL, wg_d2 = 0uL;
        for (uint i = 0; i < simd_count; ++i) {
            wg_r1 += ((ulong)sg_r1_hi[i] << 32uL) | (ulong)sg_r1_lo[i];
            wg_d1 += ((ulong)sg_d1_hi[i] << 32uL) | (ulong)sg_d1_lo[i];
            wg_r2 += ((ulong)sg_r2_hi[i] << 32uL) | (ulong)sg_r2_lo[i];
            wg_d2 += ((ulong)sg_d2_hi[i] << 32uL) | (ulong)sg_d2_lo[i];
        }
        const uint base = (bid.y * grid_groups.x + bid.x) * 4u;
        sums_lo[base + 0] = (uint)(wg_r1 & 0xFFFFFFFFuL);
        sums_hi[base + 0] = (uint)(wg_r1 >> 32uL);
        sums_lo[base + 1] = (uint)(wg_d1 & 0xFFFFFFFFuL);
        sums_hi[base + 1] = (uint)(wg_d1 >> 32uL);
        sums_lo[base + 2] = (uint)(wg_r2 & 0xFFFFFFFFuL);
        sums_hi[base + 2] = (uint)(wg_r2 >> 32uL);
        sums_lo[base + 3] = (uint)(wg_d2 & 0xFFFFFFFFuL);
        sums_hi[base + 3] = (uint)(wg_d2 >> 32uL);
    }
}
