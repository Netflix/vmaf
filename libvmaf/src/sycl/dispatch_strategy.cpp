/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  SYCL dispatch_strategy implementation. See dispatch_strategy.h
 *  and ADR-0181.
 */
#include "dispatch_strategy.h"
#include "../gpu_dispatch_parse.h"

#include <stdlib.h>

/* Backend default: SYCL graph replay wins above 720p frame area on
 * Intel Arc A380 / oneAPI 2025.3 (empirical sweep documented in
 * libvmaf/src/sycl/common.cpp § "Resolution-aware default").
 * Smaller frames pay more in graph setup than they save in
 * dispatch overhead. */
#define VMAF_SYCL_DEFAULT_AREA_THRESHOLD (1280U * 720U)

/* Strategy name table — index matches VmafSyclDispatchStrategy enum values:
 *   0 → VMAF_SYCL_DISPATCH_DIRECT
 *   1 → VMAF_SYCL_DISPATCH_GRAPH_REPLAY
 * See ADR-0483. */
static const char *const k_sycl_strategy_names[] = {
    "direct", /* VMAF_SYCL_DISPATCH_DIRECT       = 0 */
    "graph",  /* VMAF_SYCL_DISPATCH_GRAPH_REPLAY = 1 */
    nullptr,
};

VmafSyclDispatchStrategy vmaf_sycl_select_strategy(const char *feature_name,
                                                   const VmafFeatureCharacteristics *chars,
                                                   unsigned frame_w, unsigned frame_h)
{
    /* Per-feature env override has highest precedence. */
    const char *env_disp = getenv("VMAF_SYCL_DISPATCH");
    int idx = static_cast<int>(VMAF_SYCL_DISPATCH_DIRECT);
    if (vmaf_gpu_dispatch_parse_env(env_disp, feature_name, k_sycl_strategy_names, &idx))
        return static_cast<VmafSyclDispatchStrategy>(idx);

    /* Legacy global env knobs. USE wins over NO when both are set
     * (matches the existing libvmaf/src/sycl/common.cpp semantics). */
    const char *env_use_graph = getenv("VMAF_SYCL_USE_GRAPH");
    const char *env_no_graph = getenv("VMAF_SYCL_NO_GRAPH");
    if (env_use_graph && env_use_graph[0] == '1')
        return VMAF_SYCL_DISPATCH_GRAPH_REPLAY;
    if (env_no_graph && env_no_graph[0] == '1')
        return VMAF_SYCL_DISPATCH_DIRECT;

    /* Descriptor-driven decision. AUTO falls through to the
     * resolution-area default. */
    if (chars) {
        if (chars->dispatch_hint == VMAF_FEATURE_DISPATCH_BATCHED)
            return VMAF_SYCL_DISPATCH_GRAPH_REPLAY;
        if (chars->dispatch_hint == VMAF_FEATURE_DISPATCH_DIRECT)
            return VMAF_SYCL_DISPATCH_DIRECT;
    }

    /* Backend default — area-threshold heuristic preserved from the
     * pre-T7-26 inline logic so behaviour at ≥720p / <720p is
     * unchanged for AUTO descriptors. */
    const unsigned threshold = (chars && chars->min_useful_frame_area) ?
                                   chars->min_useful_frame_area :
                                   VMAF_SYCL_DEFAULT_AREA_THRESHOLD;
    const unsigned long area = (unsigned long)frame_w * (unsigned long)frame_h;
    return (area >= threshold) ? VMAF_SYCL_DISPATCH_GRAPH_REPLAY : VMAF_SYCL_DISPATCH_DIRECT;
}
