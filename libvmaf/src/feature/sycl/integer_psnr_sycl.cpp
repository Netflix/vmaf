/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  PSNR feature extractor on the SYCL backend (T7-23 / ADR-0182,
 *  GPU long-tail batch 1b part 2; chroma extension T3-15(b) second
 *  port, 2026-05-09 — see research digest
 *  `docs/research/0090-t3-15-gpu-coverage-long-tail-2026-05-09.md`,
 *  Vulkan precedent in [ADR-0216](../../docs/adr/0216-vulkan-chroma-psnr.md),
 *  CUDA twin in PR #520 / commit 7f3d58a5).
 *
 *  Algorithm (mirrors libvmaf/src/feature/integer_psnr.c::sse_line_{8,16}):
 *      diff = (int64)ref - (int64)dis;     (per pixel)
 *      sse  += diff * diff;                (atomic int64 reduction)
 *
 *  One SSE reduction per active plane (Y, Cb, Cr) per frame; the
 *  same plane-agnostic kernel is invoked three times against per-
 *  plane (w, h) and per-plane device buffers. Chroma buffers are
 *  sized per the active subsampling (4:2:0 → w/2 × h/2,
 *  4:2:2 → w/2 × h, 4:4:4 → w × h). YUV400 clamps `n_planes = 1`.
 *
 *  Buffer layout differs from luma: luma reads from the SYCL state's
 *  shared frame buffer (`vmaf_sycl_shared_frame_init`, set up
 *  luma-only by design — see `libvmaf/src/sycl/common.h`). Chroma
 *  rides on per-extractor device buffers populated by host-side
 *  staging copies in `pre_fn` (the parallel pattern used by
 *  `float_psnr_sycl.cpp`). Direct enqueue on the combined queue
 *  preserves in-order ordering with the graph-replayed luma kernel.
 *
 *  Phases (combined-graph contract — see `vmaf_sycl_graph_register`
 *  docs in `libvmaf/src/sycl/common.h`):
 *      pre_fn   : zero all 3 SSE accumulators + H2D copy chroma planes.
 *      enqueue  : luma SSE reduction kernel (graph-recordable).
 *      post_fn  : chroma SSE reduction kernels (direct) + D2H all 3
 *                 SSE accumulators.
 *
 *  Pattern: register with vmaf_sycl_graph_register and ride the
 *  combined-graph submit/wait machinery just like motion_sycl.
 */

#include <sycl/sycl.hpp>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "picture.h"
#include "sycl/common.h"
}

namespace
{

constexpr unsigned PSNR_NUM_PLANES = 3U;

struct PsnrStateSycl {
    /* Per-plane frame geometry. Plane 0 = luma. */
    unsigned width[PSNR_NUM_PLANES];
    unsigned height[PSNR_NUM_PLANES];
    unsigned bpc;
    uint32_t peak;
    /* Per-plane psnr_max — `(6 * bpc) + 12` in the default branch
     * (CPU integer_psnr.c::init's `min_sse == 0.0` path). The array
     * layout leaves `min_sse`-driven per-plane formulas a one-line
     * change away. */
    double psnr_max[PSNR_NUM_PLANES];
    /* `enable_chroma` option: when false, only luma is dispatched.
     * Default true mirrors CPU integer_psnr.c — see ADR-0453. */
    bool enable_chroma;
    /* Number of active planes (1 for YUV400, 3 otherwise). */
    unsigned n_planes;

    /* SYCL state back-pointer. */
    VmafSyclState *sycl_state;

    /* Per-plane device + host SSE accumulators. Plane 0 is the luma
     * accumulator written by the graph-recorded enqueue kernel;
     * planes 1/2 are written by chroma kernels in post_fn. */
    int64_t *d_sse[PSNR_NUM_PLANES];
    int64_t *h_sse[PSNR_NUM_PLANES];

    /* Per-extractor chroma device buffers (planes 1/2 only — luma
     * uses the shared frame buffer). Tightly packed at
     * `width[p] * bytes_per_pixel`. */
    void *d_chroma_ref[PSNR_NUM_PLANES];
    void *d_chroma_dis[PSNR_NUM_PLANES];
    /* Host staging buffers for the chroma H2D copies (USM host so
     * the queue.memcpy is a true async DMA, not a blocking copy). */
    void *h_chroma_ref[PSNR_NUM_PLANES];
    void *h_chroma_dis[PSNR_NUM_PLANES];
    size_t chroma_bytes[PSNR_NUM_PLANES];

    /* Submit/collect plumbing. */
    bool has_pending;
    unsigned pending_index;

    VmafDictionary *feature_name_dict;
};

/* Per-pixel SSE kernel. Reads the supplied ref/dis device buffers —
 * tightly packed at `width * bytes_per_pixel`. Atomic-adds each
 * pixel's int64 squared error to the device accumulator. Plane-
 * agnostic: callers pass the appropriate (ref, dis, accumulator,
 * width, height) tuple. */
static void launch_sse(sycl::queue &q, const void *ref_buf, const void *dis_buf, int64_t *d_sse,
                       unsigned width, unsigned height, unsigned bpc)
{
    sycl::range<2> global{(size_t)height, (size_t)width};
    const unsigned e_w = width;
    const unsigned e_bpc = bpc;
    const void *ref_in = ref_buf;
    const void *dis_in = dis_buf;

    q.submit([=](sycl::handler &h) {
        h.parallel_for(global, [=](sycl::id<2> id) {
            const size_t y = id[0];
            const size_t x = id[1];
            const size_t off = y * (size_t)e_w + x;
            int64_t r;
            int64_t d;
            if (e_bpc <= 8) {
                r = (int64_t)static_cast<const uint8_t *>(ref_in)[off];
                d = (int64_t)static_cast<const uint8_t *>(dis_in)[off];
            } else {
                r = (int64_t)static_cast<const uint16_t *>(ref_in)[off];
                d = (int64_t)static_cast<const uint16_t *>(dis_in)[off];
            }
            const int64_t diff = r - d;
            const int64_t se = diff * diff;
            sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                accum(*d_sse);
            accum.fetch_add(se);
        });
    });
}

/* Stage one chroma plane from a VmafPicture into a tightly-packed
 * host buffer. Mirrors `float_psnr_sycl.cpp::copy_y_plane`. */
template <typename T>
static void stage_chroma_plane(const VmafPicture *pic, unsigned plane, void *dst, unsigned w,
                               unsigned h)
{
    const T *src = static_cast<const T *>(pic->data[plane]);
    T *out = static_cast<T *>(dst);
    const ptrdiff_t src_stride_t = pic->stride[plane] / static_cast<ptrdiff_t>(sizeof(T));
    for (unsigned i = 0; i < h; i++) {
        for (unsigned j = 0; j < w; j++)
            out[j] = src[j];
        src += src_stride_t;
        out += w;
    }
}

/* The submit-side picture pointers are captured here so pre_fn /
 * post_fn (which run on the combined queue but see only the priv
 * state) can stage chroma. submit_fex_sycl writes them; pre_fn
 * reads them; the in-order combined queue serializes against
 * graph_submit so this is single-threaded per frame. */
struct PendingPics {
    VmafPicture *ref;
    VmafPicture *dis;
};

/* Pre-graph: zero the SSE accumulators + H2D copy chroma planes
 * (direct enqueue, outside graph). */
static void psnr_pre_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<PsnrStateSycl *>(priv);
    for (unsigned p = 0; p < s->n_planes; p++) {
        q.memset(s->d_sse[p], 0, sizeof(int64_t));
    }
    /* Chroma H2D — host staging buffer was populated in submit. */
    for (unsigned p = 1; p < s->n_planes; p++) {
        q.memcpy(s->d_chroma_ref[p], s->h_chroma_ref[p], s->chroma_bytes[p]);
        q.memcpy(s->d_chroma_dis[p], s->h_chroma_dis[p], s->chroma_bytes[p]);
    }
}

/* Graph-recorded: the luma per-pixel reduction kernel. Chroma stays
 * out of the graph — its inputs depend on per-frame H2D copies that
 * the L0 graph runtime cannot reliably replay (same constraint that
 * keeps memcpy/memset out per `common.h`). */
static void enqueue_psnr_work(void *queue_ptr, void *priv, void *shared_ref, void *shared_dis)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<PsnrStateSycl *>(priv);
    launch_sse(q, shared_ref, shared_dis, s->d_sse[0], s->width[0], s->height[0], s->bpc);
}

/* Post-graph: chroma SSE kernels (direct, post-graph) + D2H copy of
 * all SSE accumulators. The combined queue is in-order, so chroma
 * kernels see the H2D copies from pre_fn, and the D2H sees the
 * luma kernel from the graph + chroma kernels above. */
static void psnr_post_graph(void *queue_ptr, void *priv)
{
    sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
    auto *s = static_cast<PsnrStateSycl *>(priv);
    for (unsigned p = 1; p < s->n_planes; p++) {
        launch_sse(q, s->d_chroma_ref[p], s->d_chroma_dis[p], s->d_sse[p], s->width[p],
                   s->height[p], s->bpc);
    }
    for (unsigned p = 0; p < s->n_planes; p++) {
        q.memcpy(s->h_sse[p], s->d_sse[p], sizeof(int64_t));
    }
}

/* No per-slot config — psnr is stateless across frames. */
static void config_psnr_slot(void *priv, int slot)
{
    (void)priv;
    (void)slot;
}

} /* anonymous namespace */

extern "C" {

static const VmafOption options_psnr_sycl[] = {{
                                                   .name = "enable_chroma",
                                                   .help = "enable calculation for chroma channels",
                                                   .offset = offsetof(PsnrStateSycl, enable_chroma),
                                                   .type = VMAF_OPT_TYPE_BOOL,
                                                   .default_val.b = true,
                                               },
                                               {0}};

static int init_fex_sycl(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc,
                         unsigned w, unsigned h)
{
    auto *s = static_cast<PsnrStateSycl *>(fex->priv);

    /* Per-plane geometry derived from pix_fmt. CPU reference:
     * libvmaf/src/feature/integer_psnr.c::init computes the same
     * (ss_hor, ss_ver) split. YUV400 has chroma absent, so
     * n_planes = 1. */
    s->width[0] = w;
    s->height[0] = h;
    if (pix_fmt == VMAF_PIX_FMT_YUV400P) {
        s->n_planes = 1U;
        s->width[1] = s->width[2] = 0U;
        s->height[1] = s->height[2] = 0U;
    } else {
        s->n_planes = PSNR_NUM_PLANES;
        const int ss_hor = (pix_fmt != VMAF_PIX_FMT_YUV444P);
        const int ss_ver = (pix_fmt == VMAF_PIX_FMT_YUV420P);
        const unsigned cw = ss_hor ? ((w + 1U) >> 1) : w;
        const unsigned ch = ss_ver ? ((h + 1U) >> 1) : h;
        s->width[1] = s->width[2] = cw;
        s->height[1] = s->height[2] = ch;
    }
    /* Mirror CPU integer_psnr.c::init's enable_chroma guard (ADR-0453):
     * when the caller passes enable_chroma=false, skip chroma dispatches
     * identically to the YUV400 path above. YUV400 already forces
     * n_planes=1, so this only activates for 4:2:0/4:2:2/4:4:4. */
    if (!s->enable_chroma && s->n_planes > 1U) {
        s->n_planes = 1U;
        s->width[1] = s->width[2] = 0U;
        s->height[1] = s->height[2] = 0U;
    }

    s->bpc = bpc;
    s->peak = (1u << bpc) - 1u;
    /* Match CPU integer_psnr.c::init's psnr_max default branch
     * (`min_sse == 0.0`): psnr_max[p] = (6 * bpc) + 12. */
    for (unsigned p = 0; p < PSNR_NUM_PLANES; p++)
        s->psnr_max[p] = (double)(6U * bpc) + 12.0;

    if (!fex->sycl_state) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_sycl: no SYCL state\n");
        return -EINVAL;
    }

    VmafSyclState *state = fex->sycl_state;
    s->sycl_state = state;

    /* Luma rides on the shared frame buffer. */
    int const err = vmaf_sycl_shared_frame_init(state, w, h, bpc);
    if (err)
        return err;

    /* Per-plane SSE accumulators. */
    for (unsigned p = 0; p < s->n_planes; p++) {
        s->d_sse[p] = static_cast<int64_t *>(vmaf_sycl_malloc_device(state, sizeof(int64_t)));
        s->h_sse[p] = static_cast<int64_t *>(vmaf_sycl_malloc_host(state, sizeof(int64_t)));
        if (!s->d_sse[p] || !s->h_sse[p]) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_sycl: SSE accumulator alloc failed\n");
            return -ENOMEM;
        }
    }

    /* Per-extractor chroma device + host staging buffers (planes 1/2). */
    const size_t bpp = (bpc <= 8) ? 1u : 2u;
    for (unsigned p = 1; p < s->n_planes; p++) {
        s->chroma_bytes[p] = (size_t)s->width[p] * s->height[p] * bpp;
        s->d_chroma_ref[p] = vmaf_sycl_malloc_device(state, s->chroma_bytes[p]);
        s->d_chroma_dis[p] = vmaf_sycl_malloc_device(state, s->chroma_bytes[p]);
        s->h_chroma_ref[p] = vmaf_sycl_malloc_host(state, s->chroma_bytes[p]);
        s->h_chroma_dis[p] = vmaf_sycl_malloc_host(state, s->chroma_bytes[p]);
        if (!s->d_chroma_ref[p] || !s->d_chroma_dis[p] || !s->h_chroma_ref[p] ||
            !s->h_chroma_dis[p]) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "psnr_sycl: chroma buffer alloc failed\n");
            return -ENOMEM;
        }
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    s->has_pending = false;

    int const err2 = vmaf_sycl_graph_register(state, enqueue_psnr_work, psnr_pre_graph,
                                              psnr_post_graph, config_psnr_slot, s, "PSNR");
    if (err2)
        return err2;

    return 0;
}

static int submit_fex_sycl(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                           VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index)
{
    (void)ref_pic_90;
    (void)dist_pic_90;

    auto *s = static_cast<PsnrStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    /* Stage chroma planes into the host buffers BEFORE graph_submit
     * — pre_fn runs the H2D copy from these buffers, and the
     * combined queue's in-order semantics guarantee staging happens
     * before any GPU consumer of the host buffer fires. */
    for (unsigned p = 1; p < s->n_planes; p++) {
        if (s->bpc <= 8) {
            stage_chroma_plane<uint8_t>(ref_pic, p, s->h_chroma_ref[p], s->width[p], s->height[p]);
            stage_chroma_plane<uint8_t>(dist_pic, p, s->h_chroma_dis[p], s->width[p], s->height[p]);
        } else {
            stage_chroma_plane<uint16_t>(ref_pic, p, s->h_chroma_ref[p], s->width[p], s->height[p]);
            stage_chroma_plane<uint16_t>(dist_pic, p, s->h_chroma_dis[p], s->width[p],
                                         s->height[p]);
        }
    }

    int const err = vmaf_sycl_graph_submit(state);
    if (err)
        return err;

    s->pending_index = index;
    s->has_pending = true;
    return 0;
}

/* psnr_name[p] — same array as the CPU path
 * (libvmaf/src/feature/integer_psnr.c::psnr_name). */
static const char *const psnr_name[PSNR_NUM_PLANES] = {"psnr_y", "psnr_cb", "psnr_cr"};

static int collect_fex_sycl(VmafFeatureExtractor *fex, unsigned index,
                            VmafFeatureCollector *feature_collector)
{
    auto *s = static_cast<PsnrStateSycl *>(fex->priv);
    VmafSyclState *state = fex->sycl_state;

    vmaf_sycl_graph_wait(state);

    int rc = 0;
    for (unsigned p = 0; p < s->n_planes; p++) {
        const double sse = (double)*s->h_sse[p];
        const double n_pixels = (double)s->width[p] * (double)s->height[p];
        const double mse = sse / n_pixels;
        /* Match CPU integer_psnr.c::extract — clamp at psnr_max[p]
         * via MIN(10*log10(peak^2 / max(mse, 1e-16)), psnr_max[p]).
         * The 1e-16 floor guards against sse == 0 (trivially identical
         * frames); the CPU path uses the same constant. */
        const double peak_sq = (double)s->peak * (double)s->peak;
        const double mse_clamped = (mse > 1e-16) ? mse : 1e-16;
        double psnr = 10.0 * std::log10(peak_sq / mse_clamped);
        if (psnr > s->psnr_max[p])
            psnr = s->psnr_max[p];

        const int e = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, psnr_name[p], psnr, index);
        if (e && rc == 0)
            rc = e;
    }
    return rc;
}

static int close_fex_sycl(VmafFeatureExtractor *fex)
{
    auto *s = static_cast<PsnrStateSycl *>(fex->priv);
    if (s->sycl_state) {
        for (unsigned p = 0; p < PSNR_NUM_PLANES; p++) {
            if (s->d_sse[p])
                vmaf_sycl_free(s->sycl_state, s->d_sse[p]);
            if (s->h_sse[p])
                vmaf_sycl_free(s->sycl_state, s->h_sse[p]);
            if (s->d_chroma_ref[p])
                vmaf_sycl_free(s->sycl_state, s->d_chroma_ref[p]);
            if (s->d_chroma_dis[p])
                vmaf_sycl_free(s->sycl_state, s->d_chroma_dis[p]);
            if (s->h_chroma_ref[p])
                vmaf_sycl_free(s->sycl_state, s->h_chroma_ref[p]);
            if (s->h_chroma_dis[p])
                vmaf_sycl_free(s->sycl_state, s->h_chroma_dis[p]);
        }
    }
    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

/* Provided features — full luma + chroma per the chroma extension
 * (T3-15(b) second port, 2026-05-09; mirrors Vulkan ADR-0216 and
 * CUDA twin in PR #520). For YUV400 sources `init` clamps
 * `n_planes` to 1 and chroma dispatches are skipped at runtime,
 * but the static list still claims chroma so the dispatcher routes
 * `psnr_cb` / `psnr_cr` requests through the SYCL twin. */
static const char *provided_features_psnr_sycl[] = {"psnr_y", "psnr_cb", "psnr_cr", NULL};

extern "C" VmafFeatureExtractor vmaf_fex_psnr_sycl = {
    .name = "psnr_sycl",
    .init = init_fex_sycl,
    .extract = NULL,
    .flush = NULL,
    .close = close_fex_sycl,
    .submit = submit_fex_sycl,
    .collect = collect_fex_sycl,
    .options = options_psnr_sycl,
    .priv_size = sizeof(PsnrStateSycl),
    .flags = VMAF_FEATURE_EXTRACTOR_SYCL,
    .provided_features = provided_features_psnr_sycl,
    /* 3 dispatches/frame (one per plane), reduction-dominated;
     * AUTO + 1080p area matches motion's profile (see ADR-0181 /
     * ADR-0182). Three small dispatches are still well under the
     * threshold where batching pays off vs. AUTO scheduling. */
    .chars =
        {
            .n_dispatches_per_frame = 3,
            .is_reduction_only = true,
            .min_useful_frame_area = 1920U * 1080U,
            .dispatch_hint = VMAF_FEATURE_DISPATCH_AUTO,
        },
};

} /* extern "C" */
