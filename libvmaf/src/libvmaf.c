/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Provide a dummy fallback for older glibc versions (< 2.32) that lack
 * __libc_single_threaded.  The weak attribute lets the real glibc symbol
 * take precedence when available. */
#ifdef __linux__
/* The name is dictated by glibc's ABI and must match exactly. */
/* NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,misc-use-internal-linkage) */
__attribute__((weak)) char __libc_single_threaded = 1;
#endif

#include "libvmaf/libvmaf.h"
#include "libvmaf/feature.h"
#include "libvmaf/picture.h"

#include "cpu.h"
#include "dnn/dnn_ctx.h"
#include "dnn/tensor_io.h"
#include "feature/feature_extractor.h"
#include "feature/feature_collector.h"
#include "metadata_handler.h"
#include "fex_ctx_vector.h"
#include "log.h"
#include "model.h"
#include "output.h"
#include "picture.h"
#include "predict.h"
#include "thread_pool.h"
#include "vcs_version.h"

#ifdef HAVE_CUDA
#include "libvmaf/libvmaf_cuda.h"

#include "cuda/common.h"
#include "cuda/cuda_helper.cuh"
#include "cuda/picture_cuda.h"
#include "cuda/ring_buffer.h"
#endif

#ifdef VMAF_PICTURE_POOL
#include "picture_pool.h"
#endif

#ifdef HAVE_SYCL
#include "libvmaf/libvmaf_sycl.h"
#include "sycl/common.h"
#endif

typedef struct VmafContext {
    VmafConfiguration cfg;
    VmafFeatureCollector *feature_collector;
    RegisteredFeatureExtractors registered_feature_extractors;
    VmafFeatureExtractorContextPool *fex_ctx_pool;
    VmafThreadPool *thread_pool;
    VmafFrameSyncContext *framesync;
#ifdef VMAF_PICTURE_POOL
    VmafPicturePool *picture_pool;
#endif
#ifdef HAVE_CUDA
    struct {
        struct {
            struct {
                unsigned w, h;
                unsigned bpc;
                enum VmafPixelFormat pix_fmt;
            } pic_params;
            enum VmafCudaPicturePreallocationMethod pic_prealloc_method;
            int device_id;
            int stream_priority;
        } cfg;
        VmafCudaState state;
        VmafCudaCookie cookie;
        VmafRingBuffer* ring_buffer;
    } cuda;
#endif
#ifdef HAVE_SYCL
    struct {
        VmafSyclState *state;
    } sycl;
#endif
    struct {
        unsigned w, h;
        enum VmafPixelFormat pix_fmt;
        unsigned bpc;
        enum VmafPictureBufferType buf_type;
    } pic_params;
    unsigned pic_cnt;
    bool flushed;
    VmafPicture prev_ref; // previous ref pic for PREV_REF extractors (in-order only)
    struct {
        VmafOrtSession *sess;
        VmafModelSidecar meta;
        bool  has_sidecar;
        int   expected_w;
        int   expected_h;
        float *in_buf;          /* size: expected_w * expected_h floats */
        size_t in_elements;
        char  *feature_name;    /* owned; published via feature_collector */
    } dnn;
} VmafContext;

#ifdef VMAF_BATCH_THREADING
typedef struct BatchThreadData {
    VmafFeatureExtractorContext **fex_ctx;
    unsigned cnt;
} BatchThreadData;

static void batch_thread_data_free(void *data)
{
    BatchThreadData *td = data;
    for (unsigned i = 0; i < td->cnt; i++) {
        if (td->fex_ctx[i]) {
            vmaf_feature_extractor_context_close(td->fex_ctx[i]);
            vmaf_feature_extractor_context_destroy(td->fex_ctx[i]);
        }
    }
    free(td->fex_ctx);
    free(td);
}
#endif

int vmaf_init(VmafContext **vmaf, VmafConfiguration cfg)
{
    if (!vmaf) return -EINVAL;
    int err = 0;

    VmafContext *const v = *vmaf = malloc(sizeof(*v));
    if (!v) goto fail;
    memset(v, 0, sizeof(*v));
    v->cfg = cfg;

    vmaf_init_cpu();
    vmaf_set_cpu_flags_mask(~cfg.cpumask);

    vmaf_set_log_level(cfg.log_level);

    err = vmaf_framesync_init(&(v->framesync));
    if (err) goto free_v;
    err = vmaf_feature_collector_init(&(v->feature_collector));
    if (err) goto free_framesync;
    err = feature_extractor_vector_init(&(v->registered_feature_extractors));
    if (err) goto free_feature_collector;

    if (v->cfg.n_threads > 0) {
        VmafThreadPoolConfig tpool_cfg = {
            .n_threads = v->cfg.n_threads,
#ifdef VMAF_BATCH_THREADING
            .thread_data_free = batch_thread_data_free,
#endif
        };
        err = vmaf_thread_pool_create(&v->thread_pool, tpool_cfg);
        if (err) goto free_feature_extractor_vector;
        err = vmaf_fex_ctx_pool_create(&v->fex_ctx_pool, v->cfg.n_threads);
        if (err) goto free_thread_pool;
    }

    return 0;

free_thread_pool:
    vmaf_thread_pool_destroy(v->thread_pool);
free_feature_extractor_vector:
    feature_extractor_vector_destroy(&(v->registered_feature_extractors));
free_feature_collector:
    vmaf_feature_collector_destroy(v->feature_collector);
free_framesync:
    vmaf_framesync_destroy(v->framesync);
free_v:
    free(v);
fail:
    return -ENOMEM;
}

#ifdef HAVE_CUDA
static int prepare_ring_buffer(VmafContext *vmaf, unsigned w, unsigned h,
                               enum VmafPixelFormat pix_fmt, unsigned bpc)
{
    if (!vmaf) return -EINVAL;
    if (!w) return -EINVAL;
    if (!h) return -EINVAL;
    if (!pix_fmt) return -EINVAL;
    if (!bpc) return -EINVAL;

    vmaf->cuda.cookie.pix_fmt = vmaf->pic_params.pix_fmt = pix_fmt;
    vmaf->cuda.cookie.h = vmaf->pic_params.h = h;
    vmaf->cuda.cookie.w = vmaf->pic_params.w = w;
    vmaf->cuda.cookie.bpc = vmaf->pic_params.bpc = bpc;
    vmaf->cuda.cookie.state = &vmaf->cuda.state;

    VmafRingBufferConfig cfg_buf = {
        .pic_cnt = 4,
        .cookie = &vmaf->cuda.cookie,
        .synchronize_picture_callback = vmaf_cuda_picture_synchronize,
        .alloc_picture_callback = vmaf_cuda_picture_alloc,
        .free_picture_callback = vmaf_cuda_picture_free,
    };

    return vmaf_ring_buffer_init(&vmaf->cuda.ring_buffer, cfg_buf);
}

int vmaf_cuda_import_state(VmafContext *vmaf, VmafCudaState *cu_state)
{
    if (!vmaf) return -EINVAL;
    if (!cu_state) return -EINVAL;

    vmaf->cuda.state = *cu_state;

    return 0;
}

int vmaf_cuda_preallocate_pictures(VmafContext *vmaf, VmafCudaPictureConfiguration cfg)
{
    if (!vmaf) return -EINVAL;

    int err = 0;

    vmaf->cuda.cfg.pic_params.w = cfg.pic_params.w;
    vmaf->cuda.cfg.pic_params.h = cfg.pic_params.h;
    vmaf->cuda.cfg.pic_params.bpc = cfg.pic_params.bpc;
    vmaf->cuda.cfg.pic_params.pix_fmt = cfg.pic_params.pix_fmt;
    vmaf->cuda.cfg.pic_prealloc_method = cfg.pic_prealloc_method;

    switch (cfg.pic_prealloc_method) {
    case VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_NONE:
        break;
    case VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST:
    case VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED:
    case VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE:
        err = prepare_ring_buffer(vmaf, cfg.pic_params.w,
                                  cfg.pic_params.h, cfg.pic_params.pix_fmt,
                                  cfg.pic_params.bpc);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem during cuda picture preallocation\n");
            return err;
        }
        break;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "unknown cuda picture preallocation method\n");
        return -EINVAL;
    }

    return err;
}

int vmaf_cuda_fetch_preallocated_picture(VmafContext *vmaf, VmafPicture* pic)
{
    if (!vmaf) return -EINVAL;
    if (!pic) return -EINVAL;
    if (!vmaf->cuda.ring_buffer) return -EINVAL;

    //TODO: preallocate host pics

    switch (vmaf->cuda.cfg.pic_prealloc_method) {
    case VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE:
        return vmaf_ring_buffer_fetch_next_picture(vmaf->cuda.ring_buffer, pic);
    case VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST:
        return vmaf_picture_alloc(pic, vmaf->cuda.cfg.pic_params.pix_fmt,
                vmaf->cuda.cfg.pic_params.bpc, vmaf->cuda.cfg.pic_params.w,
                vmaf->cuda.cfg.pic_params.h);
    case VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED:
        return vmaf_cuda_picture_alloc_pinned(pic, vmaf->cuda.cfg.pic_params.pix_fmt,
                vmaf->cuda.cfg.pic_params.bpc, vmaf->cuda.cfg.pic_params.w,
                vmaf->cuda.cfg.pic_params.h, &vmaf->cuda.state);
    case VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_NONE:
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "undefined cuda picture preallocation method\n");
        return -EINVAL;
    }
}

static int set_fex_cuda_state(VmafFeatureExtractorContext *fex_ctx,
                              VmafContext *vmaf)
{
    if (fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA)
        fex_ctx->fex->cu_state = &(vmaf->cuda.state);
    return 0;
}

#endif

#ifdef VMAF_PICTURE_POOL
static int prepare_picture_pool(VmafContext *vmaf, unsigned pic_cnt,
                                unsigned w, unsigned h,
                                enum VmafPixelFormat pix_fmt, unsigned bpc)
{
    if (!vmaf) return -EINVAL;
    if (!w || !h) return -EINVAL;
    if (!pic_cnt) return -EINVAL;

    VmafPicturePoolConfig cfg = {
        .pic_cnt = pic_cnt,
        .w = w,
        .h = h,
        .pix_fmt = pix_fmt,
        .bpc = bpc,
    };

    return vmaf_picture_pool_init(&vmaf->picture_pool, cfg);
}

static int check_picture_pool(VmafContext *vmaf)
{
    if (!vmaf->thread_pool) return 0;
    if (vmaf->picture_pool) return 0;

    // Default to 2x thread count if not explicitly preallocated
    const unsigned pic_cnt = vmaf->cfg.n_threads * 2;

    int err = prepare_picture_pool(vmaf, pic_cnt,
                                   vmaf->pic_params.w,
                                   vmaf->pic_params.h,
                                   vmaf->pic_params.pix_fmt,
                                   vmaf->pic_params.bpc);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "problem during prepare_picture_pool\n");
        return -EINVAL;
    }

    return 0;
}

int vmaf_preallocate_pictures(VmafContext *vmaf,
                                   VmafPictureConfiguration cfg)
{
    if (!vmaf) return -EINVAL;

    return prepare_picture_pool(vmaf, cfg.pic_cnt,
                                cfg.pic_params.w, cfg.pic_params.h,
                                cfg.pic_params.pix_fmt, cfg.pic_params.bpc);
}

int vmaf_fetch_preallocated_picture(VmafContext *vmaf, VmafPicture *pic)
{
    if (!vmaf) return -EINVAL;
    if (!pic) return -EINVAL;
    if (!vmaf->picture_pool) return -EINVAL;

    return vmaf_picture_pool_fetch(vmaf->picture_pool, pic);
}
#endif

#ifdef HAVE_SYCL
int vmaf_sycl_import_state(VmafContext *vmaf, VmafSyclState *sycl_state)
{
    if (!vmaf) return -EINVAL;
    if (!sycl_state) return -EINVAL;

    vmaf->sycl.state = sycl_state;

    return 0;
}

int vmaf_sycl_preallocate_pictures(VmafContext *vmaf,
                                    VmafSyclPictureConfiguration cfg)
{
    // SYCL extractors handle picture upload internally,
    // so preallocation is not strictly needed.
    (void)vmaf; (void)cfg;
    return 0;
}

int vmaf_sycl_picture_fetch(VmafContext *vmaf, VmafPicture *pic)
{
    if (!vmaf) return -EINVAL;
    if (!pic) return -EINVAL;

    return vmaf_picture_alloc(pic, vmaf->pic_params.pix_fmt,
                              vmaf->pic_params.bpc,
                              vmaf->pic_params.w, vmaf->pic_params.h);
}

int vmaf_sycl_init_frame_buffers(VmafContext *vmaf,
                                  unsigned w, unsigned h, unsigned bpc)
{
    if (!vmaf) return -EINVAL;
    if (!vmaf->sycl.state) return -EINVAL;

    vmaf->pic_params.w = w;
    vmaf->pic_params.h = h;
    vmaf->pic_params.bpc = bpc;
    vmaf->pic_params.pix_fmt = VMAF_PIX_FMT_YUV420P;

    return vmaf_sycl_shared_frame_init(vmaf->sycl.state, w, h, bpc);
}

int vmaf_sycl_get_frame_buffers(VmafContext *vmaf,
                                 void **ref, void **dis)
{
    if (!vmaf || !vmaf->sycl.state) return -EINVAL;

    return vmaf_sycl_shared_frame_get(vmaf->sycl.state, ref, dis);
}

int vmaf_sycl_wait_compute(VmafContext *vmaf)
{
    if (!vmaf) return -EINVAL;
    if (!vmaf->sycl.state) return -EINVAL;

    // Wait on the primary queue (VA surface imports / de-tile kernels)
    int err = vmaf_sycl_queue_wait(vmaf->sycl.state);
    if (err) return err;

    // Also wait on the combined compute queue (GPU feature extractors).
    // Without this, the VA import path can overwrite shared ref/dis buffers
    // while the previous frame's extractors are still reading them.
    return vmaf_sycl_combined_queue_wait(vmaf->sycl.state);
}

static int set_fex_sycl_state(VmafFeatureExtractorContext *fex_ctx,
                               VmafContext *vmaf)
{
    if (fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_SYCL)
        fex_ctx->fex->sycl_state = vmaf->sycl.state;
    return 0;
}
#endif

static int set_fex_framesync(VmafFeatureExtractorContext *fex_ctx,
                              VmafContext *vmaf)
{
    if (fex_ctx->fex->flags & VMAF_FEATURE_FRAME_SYNC)
        fex_ctx->fex->framesync = (vmaf->framesync);
    return 0;
}

static void vmaf_ctx_dnn_free(VmafContext *vmaf)
{
    if (!vmaf) return;
    if (vmaf->dnn.sess) {
        vmaf_ort_close(vmaf->dnn.sess);
        vmaf->dnn.sess = NULL;
    }
    if (vmaf->dnn.has_sidecar) {
        vmaf_dnn_sidecar_free(&vmaf->dnn.meta);
        vmaf->dnn.has_sidecar = false;
    }
    free(vmaf->dnn.in_buf);
    vmaf->dnn.in_buf = NULL;
    vmaf->dnn.in_elements = 0;
    free(vmaf->dnn.feature_name);
    vmaf->dnn.feature_name = NULL;
}

int vmaf_ctx_dnn_has_session(const VmafContext *ctx)
{
    return (ctx && ctx->dnn.sess) ? 1 : 0;
}

int vmaf_ctx_dnn_attach(VmafContext *ctx,
                        VmafOrtSession *sess,
                        const VmafModelSidecar *meta,
                        const int64_t *in_shape, size_t in_rank,
                        const char *feature_name)
{
    if (!ctx || !sess || !in_shape || !feature_name) return -EINVAL;
    if (ctx->dnn.sess) return -EBUSY;

    /* Only NCHW [1, 1, H, W] is supported for the current wiring. Anything
     * else is a hard -ENOTSUP so users see the limit rather than silent
     * mis-inference. */
    if (in_rank != 4) return -ENOTSUP;
    if (in_shape[0] != 1 || in_shape[1] != 1) return -ENOTSUP;
    const int64_t h = in_shape[2];
    const int64_t w = in_shape[3];
    if (h <= 0 || w <= 0) return -ENOTSUP;   /* dynamic dims unsupported */

    const size_t n = (size_t) w * (size_t) h;
    float *buf = (float *) calloc(n, sizeof(*buf));
    if (!buf) return -ENOMEM;

    char *name = strdup(feature_name);
    if (!name) { free(buf); return -ENOMEM; }

    ctx->dnn.sess         = sess;
    if (meta) {
        ctx->dnn.meta        = *meta;
        ctx->dnn.has_sidecar = true;
    }
    ctx->dnn.expected_w   = (int) w;
    ctx->dnn.expected_h   = (int) h;
    ctx->dnn.in_buf       = buf;
    ctx->dnn.in_elements  = n;
    ctx->dnn.feature_name = name;
    return 0;
}

static int vmaf_ctx_dnn_run_frame(VmafContext *vmaf, VmafPicture *ref,
                                  unsigned index)
{
    if (!vmaf->dnn.sess) return 0;
    if (!ref || !ref->data[0]) return -EINVAL;

    /* The current tensor bridge operates on 8-bit luma only. 10/12-bit
     * content and multi-channel inputs are rejected loudly rather than
     * quietly truncated. */
    if (ref->bpc != 8) return -ENOTSUP;
    if ((int) ref->w[0] != vmaf->dnn.expected_w ||
        (int) ref->h[0] != vmaf->dnn.expected_h) {
        return -ERANGE;
    }

    const float *mean = NULL;
    const float *std  = NULL;
    float m = 0.f;
    float s = 1.f;
    if (vmaf->dnn.has_sidecar && vmaf->dnn.meta.has_norm) {
        m = vmaf->dnn.meta.norm_mean;
        s = vmaf->dnn.meta.norm_std;
        if (s > 0.f) { mean = &m; std = &s; }
    }

    int rc = vmaf_tensor_from_luma((const uint8_t *) ref->data[0],
                                   (size_t) ref->stride[0],
                                   vmaf->dnn.expected_w, vmaf->dnn.expected_h,
                                   VMAF_TENSOR_LAYOUT_NCHW,
                                   VMAF_TENSOR_DTYPE_F32,
                                   mean, std,
                                   vmaf->dnn.in_buf);
    if (rc < 0) return rc;

    const int64_t shape[4] = {
        1, 1, vmaf->dnn.expected_h, vmaf->dnn.expected_w
    };
    float out = 0.f;
    size_t out_n = 0;
    rc = vmaf_ort_infer(vmaf->dnn.sess,
                        vmaf->dnn.in_buf, shape, 4,
                        &out, 1u, &out_n);
    if (rc < 0) return rc;
    if (out_n != 1u) return -ENOTSUP;  /* multi-value outputs unsupported */

    return vmaf_feature_collector_append(vmaf->feature_collector,
                                         vmaf->dnn.feature_name,
                                         (double) out, index);
}

int vmaf_close(VmafContext *vmaf)
{
    if (!vmaf) return -EINVAL;

    vmaf_thread_pool_wait(vmaf->thread_pool);
    if (vmaf->prev_ref.ref)
        vmaf_picture_unref(&vmaf->prev_ref);
    vmaf_framesync_destroy(vmaf->framesync);
    feature_extractor_vector_destroy(&(vmaf->registered_feature_extractors));
    vmaf_feature_collector_destroy(vmaf->feature_collector);
    vmaf_thread_pool_destroy(vmaf->thread_pool);
    vmaf_fex_ctx_pool_destroy(vmaf->fex_ctx_pool);
    vmaf_ctx_dnn_free(vmaf);
#ifdef VMAF_PICTURE_POOL
    if (vmaf->picture_pool)
        vmaf_picture_pool_close(vmaf->picture_pool);
#endif
#ifdef HAVE_CUDA
    if (vmaf->cuda.ring_buffer)
        vmaf_ring_buffer_close(vmaf->cuda.ring_buffer);
    if (vmaf->cuda.state.ctx)
        vmaf_cuda_release(&vmaf->cuda.state);
#endif
#ifdef HAVE_SYCL
    /* Note: ownership of sycl.state is NOT transferred by
     * vmaf_sycl_import_state(), so we do not free it here.
     * The caller must call vmaf_sycl_state_free() after vmaf_close(). */
    vmaf->sycl.state = NULL;
#endif
    free(vmaf);

    return 0;
}

int vmaf_import_feature_score(VmafContext *vmaf, const char *feature_name,
                              double value, unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;

    return vmaf_feature_collector_append(vmaf->feature_collector, feature_name,
                                         value, index);
}

int vmaf_use_feature(VmafContext *vmaf, const char *feature_name,
                     VmafFeatureDictionary *opts_dict)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;

    VmafDictionary *s = (VmafDictionary*) opts_dict;

    int err = 0;

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name(feature_name);
    if (!fex) return -EINVAL;

    VmafDictionary *d = NULL;
    if (s) {
        err = vmaf_dictionary_copy(&s, &d);
        if (err) return err;
        err = vmaf_dictionary_free(&s);
        if (err) return err;
    }

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, d);
#ifdef HAVE_CUDA
    err |= set_fex_cuda_state(fex_ctx, vmaf);
#endif
#ifdef HAVE_SYCL
    err |= set_fex_sycl_state(fex_ctx, vmaf);
#endif
    err |= set_fex_framesync(fex_ctx, vmaf);
    if (err) return err;

    RegisteredFeatureExtractors *rfe = &(vmaf->registered_feature_extractors);
    err = feature_extractor_vector_append(rfe, fex_ctx, 0);
    if (err)
        err |= vmaf_feature_extractor_context_destroy(fex_ctx);

    return err;
}

int vmaf_use_features_from_model(VmafContext *vmaf, VmafModel *model)
{
    if (!vmaf) return -EINVAL;
    if (!model) return -EINVAL;

    int err = 0;

    unsigned fex_flags = 0;

#ifdef HAVE_CUDA
    if (!vmaf->cfg.gpumask && vmaf->cuda.state.ctx)
        fex_flags |= VMAF_FEATURE_EXTRACTOR_CUDA;
#endif
#ifdef HAVE_SYCL
    if (!vmaf->cfg.gpumask && vmaf->sycl.state)
        fex_flags |= VMAF_FEATURE_EXTRACTOR_SYCL;
#endif

    RegisteredFeatureExtractors *rfe = &(vmaf->registered_feature_extractors);

    for (unsigned i = 0; i < model->n_features; i++) {
        VmafFeatureExtractor *fex =
            vmaf_get_feature_extractor_by_feature_name(model->feature[i].name,
                                                       fex_flags);
        if (!fex) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "could not initialize feature extractor \"%s\"\n",
                     model->feature[i].name);
            return -EINVAL;
        }

        VmafFeatureExtractorContext *fex_ctx;
        VmafDictionary *d = NULL;
        if (model->feature[i].opts_dict) {
            err = vmaf_dictionary_copy(&model->feature[i].opts_dict, &d);
            if (err) return err;
        }
        err = vmaf_feature_extractor_context_create(&fex_ctx, fex, d);
#ifdef HAVE_CUDA
        err |= set_fex_cuda_state(fex_ctx, vmaf);
#endif
#ifdef HAVE_SYCL
        err |= set_fex_sycl_state(fex_ctx, vmaf);
#endif
        err |= set_fex_framesync(fex_ctx, vmaf);
        if (err) return err;
        err = feature_extractor_vector_append(rfe, fex_ctx, 0);
        if (err) {
            err |= vmaf_feature_extractor_context_destroy(fex_ctx);
            return err;
        }
    }

    err = vmaf_feature_collector_mount_model(vmaf->feature_collector, model);
    if (err) return err;

    return 0;
}

int vmaf_use_features_from_model_collection(VmafContext *vmaf,
                                            VmafModelCollection *model_collection)
{
    if (!vmaf) return -EINVAL;
    if (!model_collection) return -EINVAL;

    int err = 0;
    for (unsigned i = 0; i < model_collection->cnt; i++)
        err |= vmaf_use_features_from_model(vmaf, model_collection->model[i]);

    return err;
}

struct ThreadData {
    VmafFeatureExtractorContext *fex_ctx;
    VmafPicture ref, dist, prev_ref;
    unsigned index;
    VmafFeatureCollector *feature_collector;
    VmafFeatureExtractorContextPool *fex_ctx_pool;
    int err;
};

static void threaded_extract_func(void *e, void **thread_data)
{
    (void) thread_data;
    struct ThreadData *f = e;

    if (f->prev_ref.ref)
        f->fex_ctx->fex->prev_ref = f->prev_ref;

    f->err = vmaf_feature_extractor_context_extract(f->fex_ctx, &f->ref, NULL,
                                                    &f->dist, NULL, f->index,
                                                    f->feature_collector);

    if (f->prev_ref.ref) {
        memset(&f->fex_ctx->fex->prev_ref, 0, sizeof(f->fex_ctx->fex->prev_ref));
        vmaf_picture_unref(&f->prev_ref);
    }

    f->err = vmaf_fex_ctx_pool_release(f->fex_ctx_pool, f->fex_ctx);
    vmaf_picture_unref(&f->ref);
    vmaf_picture_unref(&f->dist);
}

#ifdef VMAF_BATCH_THREADING
struct ThreadDataBatch {
    VmafPicture ref, dist, prev_ref;
    unsigned index;
    VmafFeatureCollector *feature_collector;
    RegisteredFeatureExtractors *registered_fex;
    unsigned n_subsample;
    int err;
};

static void threaded_extract_batch_func(void *e, void **thread_data)
{
    struct ThreadDataBatch *f = e;
    f->err = 0;

    BatchThreadData *td = *thread_data;
    if (!td) {
        td = malloc(sizeof(*td));
        if (!td) { f->err = -ENOMEM; goto unref; }
        td->cnt = f->registered_fex->cnt;
        td->fex_ctx = calloc(td->cnt, sizeof(*td->fex_ctx));
        if (!td->fex_ctx) { free(td); f->err = -ENOMEM; goto unref; }
        *thread_data = td;
    }

    for (unsigned i = 0; i < f->registered_fex->cnt; i++) {
        VmafFeatureExtractor *fex = f->registered_fex->fex_ctx[i]->fex;

        if (fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA)
            continue;

        if (fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL)
            continue;

        if ((f->n_subsample > 1) && (f->index % f->n_subsample))
            continue;

        if (!td->fex_ctx[i]) {
            VmafDictionary *opts_dict = f->registered_fex->fex_ctx[i]->opts_dict;
            VmafDictionary *d = NULL;
            if (opts_dict) {
                int err = vmaf_dictionary_copy(&opts_dict, &d);
                if (err) { f->err = err; break; }
            }
            int err = vmaf_feature_extractor_context_create(&td->fex_ctx[i],
                                                             fex, d);
            if (err) { f->err = err; break; }
        }

        if (fex->flags & VMAF_FEATURE_EXTRACTOR_PREV_REF) {
            if (f->prev_ref.ref)
                td->fex_ctx[i]->fex->prev_ref = f->prev_ref;
        }

        int err = vmaf_feature_extractor_context_extract(td->fex_ctx[i],
                                                         &f->ref, NULL,
                                                         &f->dist, NULL,
                                                         f->index,
                                                         f->feature_collector);

        if (fex->flags & VMAF_FEATURE_EXTRACTOR_PREV_REF)
            memset(&td->fex_ctx[i]->fex->prev_ref, 0,
                   sizeof(td->fex_ctx[i]->fex->prev_ref));

        if (err) {
            f->err = err;
            break;
        }
    }

unref:
    if (f->prev_ref.ref)
        vmaf_picture_unref(&f->prev_ref);
    vmaf_picture_unref(&f->ref);
    vmaf_picture_unref(&f->dist);
}
#endif // VMAF_BATCH_THREADING

/* NOLINTNEXTLINE(readability-function-size) */
static int threaded_read_pictures(VmafContext *vmaf, VmafPicture *ref,
                                  VmafPicture *dist, unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!ref) return -EINVAL;
    if (!dist) return -EINVAL;

    int err = 0;

    for (unsigned i = 0; i < vmaf->registered_feature_extractors.cnt; i++) {
        VmafFeatureExtractor *fex =
            vmaf->registered_feature_extractors.fex_ctx[i]->fex;
        if (fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA)
            continue;
        if (fex->flags & VMAF_FEATURE_EXTRACTOR_SYCL)
            continue;
        VmafDictionary *opts_dict =
            vmaf->registered_feature_extractors.fex_ctx[i]->opts_dict;

        if ((vmaf->cfg.n_subsample > 1) && (index % vmaf->cfg.n_subsample) &&
            !(fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL))
        {
            continue;
        }

        fex->framesync = vmaf->framesync;
        VmafFeatureExtractorContext *fex_ctx;
        err = vmaf_fex_ctx_pool_aquire(vmaf->fex_ctx_pool, fex, opts_dict,
                                       &fex_ctx);
        if (err) return err;

        VmafPicture pic_a;
        VmafPicture pic_b;
        VmafPicture prev_ref = { 0 };
        vmaf_picture_ref(&pic_a, ref);
        vmaf_picture_ref(&pic_b, dist);

        if ((fex->flags & VMAF_FEATURE_EXTRACTOR_PREV_REF) &&
            vmaf->prev_ref.ref)
        {
            vmaf_picture_ref(&prev_ref, &vmaf->prev_ref);
        }

        struct ThreadData data = {
            .fex_ctx = fex_ctx,
            .ref = pic_a,
            .dist = pic_b,
            .prev_ref = prev_ref,
            .index = index,
            .feature_collector = vmaf->feature_collector,
            .fex_ctx_pool = vmaf->fex_ctx_pool,
            .err = 0,
        };

        err = vmaf_thread_pool_enqueue(vmaf->thread_pool, threaded_extract_func,
                                       &data, sizeof(data));
        if (err) {
            vmaf_picture_unref(&pic_a);
            vmaf_picture_unref(&pic_b);
            if (prev_ref.ref) vmaf_picture_unref(&prev_ref);
            return err;
        }
    }

    if (vmaf->prev_ref.ref)
        vmaf_picture_unref(&vmaf->prev_ref);
    vmaf_picture_ref(&vmaf->prev_ref, ref);

    return vmaf_picture_unref(ref) | vmaf_picture_unref(dist);
}

#ifdef VMAF_BATCH_THREADING
static int threaded_read_pictures_batch(VmafContext *vmaf, VmafPicture *ref,
                                        VmafPicture *dist, unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!ref) return -EINVAL;
    if (!dist) return -EINVAL;

    int err = 0;

    VmafPicture pic_a, pic_b, prev_ref = { 0 };
    vmaf_picture_ref(&pic_a, ref);
    vmaf_picture_ref(&pic_b, dist);

    if (vmaf->prev_ref.ref)
        vmaf_picture_ref(&prev_ref, &vmaf->prev_ref);

    struct ThreadDataBatch data = {
        .ref = pic_a,
        .dist = pic_b,
        .prev_ref = prev_ref,
        .index = index,
        .feature_collector = vmaf->feature_collector,
        .registered_fex = &vmaf->registered_feature_extractors,
        .n_subsample = vmaf->cfg.n_subsample,
        .err = 0,
    };

    err = vmaf_thread_pool_enqueue(vmaf->thread_pool, threaded_extract_batch_func,
                                   &data, sizeof(data));
    if (err) {
        vmaf_picture_unref(&pic_a);
        vmaf_picture_unref(&pic_b);
        if (prev_ref.ref) vmaf_picture_unref(&prev_ref);
        return err;
    }

    if (vmaf->prev_ref.ref)
        vmaf_picture_unref(&vmaf->prev_ref);
    vmaf_picture_ref(&vmaf->prev_ref, ref);

    return vmaf_picture_unref(ref) | vmaf_picture_unref(dist);
}
#endif // VMAF_BATCH_THREADING

static int validate_pic_params(VmafContext *vmaf, VmafPicture *ref,
                               VmafPicture *dist)
{
    VmafPicturePrivate *ref_priv = ref->priv;
    VmafPicturePrivate *dist_priv = dist->priv;

    if (!vmaf->pic_params.w) {
        vmaf->pic_params.w = ref->w[0];
        vmaf->pic_params.h = ref->h[0];
        vmaf->pic_params.pix_fmt = ref->pix_fmt;
        vmaf->pic_params.bpc = ref->bpc;
    }
    vmaf->pic_params.buf_type = ref_priv->buf_type;

    if ((ref->w[0] != dist->w[0]) || (ref->w[0] != vmaf->pic_params.w))
        return -EINVAL;
    if ((ref->h[0] != dist->h[0]) || (ref->h[0] != vmaf->pic_params.h))
        return -EINVAL;
    if ((ref->pix_fmt != dist->pix_fmt) ||
        (ref->pix_fmt != vmaf->pic_params.pix_fmt))
    {
        return -EINVAL;
    }
    if ((ref->bpc != dist->bpc) && (ref->bpc != vmaf->pic_params.bpc))
        return -EINVAL;
    if (ref_priv->buf_type != dist_priv->buf_type)
        return -EINVAL;

    return 0;
}

static int flush_context_threaded(VmafContext *vmaf)
{
    int err = 0;
    err |= vmaf_thread_pool_wait(vmaf->thread_pool);
#ifdef VMAF_BATCH_THREADING
    RegisteredFeatureExtractors rfe = vmaf->registered_feature_extractors;
    for (unsigned i = 0; i < rfe.cnt; i++) {
        if (!(rfe.fex_ctx[i]->fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL))
            continue;
        err |= vmaf_feature_extractor_context_flush(rfe.fex_ctx[i],
                                                    vmaf->feature_collector);
    }
#else
    err |= vmaf_fex_ctx_pool_flush(vmaf->fex_ctx_pool, vmaf->feature_collector);
#endif

    {
        RegisteredFeatureExtractors rfe = vmaf->registered_feature_extractors;
        for (unsigned i = 0; i < rfe.cnt; i++) {
            VmafFeatureExtractor *fex = rfe.fex_ctx[i]->fex;
            if (fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL)
                continue;
            if (fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA)
                continue;
            if (!fex->flush)
                continue;
            int flush_err = 0;
            while (!(flush_err = fex->flush(fex, vmaf->feature_collector)));
            if (flush_err < 0) err |= flush_err;
        }
    }

    if (!err) vmaf->flushed = true;
    return err;
}

/* NOLINTNEXTLINE(readability-function-size) */
static int flush_context(VmafContext *vmaf)
{
    int err = 0;
    if (vmaf->thread_pool) {
        err = flush_context_threaded(vmaf);
    } else {
        RegisteredFeatureExtractors rfe = vmaf->registered_feature_extractors;
        for (unsigned i = 0; i < rfe.cnt; i++) {
            if (!(rfe.fex_ctx[i]->fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA) &&
                !(rfe.fex_ctx[i]->fex->flags & VMAF_FEATURE_EXTRACTOR_SYCL)) {
                err |= vmaf_feature_extractor_context_flush(rfe.fex_ctx[i],
                                                            vmaf->feature_collector);
            }
        }
    }

#ifdef HAVE_CUDA
    if (vmaf->cuda.state.ctx) {
        RegisteredFeatureExtractors rfe = vmaf->registered_feature_extractors;
        for (unsigned i = 0; i < rfe.cnt; i++) {
            if (rfe.fex_ctx[i]->fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA) {
                // Collect any pending double-buffered CUDA work
                if (rfe.fex_ctx[i]->gpu_pending) {
                    err |= vmaf_feature_extractor_context_collect(rfe.fex_ctx[i],
                            rfe.fex_ctx[i]->gpu_pending_index,
                            vmaf->feature_collector);
                    rfe.fex_ctx[i]->gpu_pending = false;
                }
                err |= vmaf_feature_extractor_context_flush(rfe.fex_ctx[i],
                                                            vmaf->feature_collector);
            }
        }
        CudaFunctions* cu_f = vmaf->cuda.state.f;
        err |= cu_f->cuCtxPushCurrent(vmaf->cuda.state.ctx);
        err |= cu_f->cuStreamSynchronize(vmaf->cuda.state.str);
        err |= cu_f->cuCtxSynchronize();
        err |= cu_f->cuCtxPopCurrent(NULL);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                    "context could not be synchronized\n");
            return -EINVAL;
        }
    }
#endif

#ifdef HAVE_SYCL
    if (vmaf->sycl.state) {
        RegisteredFeatureExtractors rfe = vmaf->registered_feature_extractors;
        // Collect any pending double-buffered SYCL work
        for (unsigned i = 0; i < rfe.cnt; i++) {
            if ((rfe.fex_ctx[i]->fex->flags & VMAF_FEATURE_EXTRACTOR_SYCL) &&
                rfe.fex_ctx[i]->gpu_pending) {
                err |= vmaf_feature_extractor_context_collect(rfe.fex_ctx[i],
                        rfe.fex_ctx[i]->gpu_pending_index,
                        vmaf->feature_collector);
                rfe.fex_ctx[i]->gpu_pending = false;
            }
        }
        for (unsigned i = 0; i < rfe.cnt; i++) {
            if (rfe.fex_ctx[i]->fex->flags & VMAF_FEATURE_EXTRACTOR_SYCL)
                err |= vmaf_feature_extractor_context_flush(rfe.fex_ctx[i],
                                                            vmaf->feature_collector);
        }
        vmaf_sycl_queue_wait(vmaf->sycl.state);
        vmaf_sycl_print_timing(vmaf->sycl.state);
    }
#endif

    if (!err) vmaf->flushed = true;
    return err;
}

#ifdef HAVE_CUDA
static int check_ring_buffer(VmafContext *vmaf)
{
    if (!vmaf->cuda.state.ctx) return 0;

    int err = 0;

    if (!vmaf->cuda.cfg.pic_prealloc_method && !vmaf->cuda.ring_buffer) {
        err = prepare_ring_buffer(vmaf, vmaf->pic_params.w, vmaf->pic_params.h,
                vmaf->pic_params.pix_fmt, vmaf->pic_params.bpc);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem during prepare_ring_buffer\n");
            return -EINVAL;
        }
    }

    return err;
}

enum  {
    HW_FLAG_HOST = 1 << 0,
    HW_FLAG_DEVICE = 1 << 1,
};

static int translate_picture_host(VmafContext *vmaf, VmafPicture *pic,
                                  VmafPicture *pic_device, unsigned hw_flags)
{
    int err = 0;
    if (!(hw_flags & HW_FLAG_DEVICE)) return err;

    //host to device

    switch(vmaf->pic_params.buf_type) {
    case VMAF_PICTURE_BUFFER_TYPE_HOST:
    case VMAF_PICTURE_BUFFER_TYPE_CUDA_HOST_PINNED:
        if (!vmaf->cuda.state.ctx)
            return -EINVAL;
        err |= vmaf_ring_buffer_fetch_next_picture(vmaf->cuda.ring_buffer, pic_device);
        err |= vmaf_cuda_picture_upload_async(pic_device, pic, 0x1);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                    "problem moving host pic into cuda device buffer\n");
            return err;
        }
        break;
    default:
        return -EINVAL;
    }

    return err;
}

static int translate_picture_device(VmafContext *vmaf, VmafPicture *pic,
                                    VmafPicture *pic_host, unsigned hw_flags)
{
    int err = 0;
    if (!(hw_flags & HW_FLAG_HOST)) return err;

    //device to host

    err = vmaf_picture_alloc(pic_host, pic->pix_fmt, pic->bpc,
                             pic->w[0], pic->h[0]);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "problem allocating host pic\n");
        return err;
    }

    err = vmaf_cuda_picture_download_async(pic, pic_host, 0x1);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "problem moving cuda pic into host buffer\n");
        return err;
    }

    return err;
}

static int translate_picture(VmafContext *vmaf, VmafPicture *pic,
                             VmafPicture *pic_host, VmafPicture *pic_device,
                             unsigned hw_flags)
{
    const VmafPicturePrivate *pic_priv = pic->priv;

    switch(pic_priv->buf_type) {
    case VMAF_PICTURE_BUFFER_TYPE_HOST:
    case VMAF_PICTURE_BUFFER_TYPE_CUDA_HOST_PINNED:
        *pic_host = *pic;
        return translate_picture_host(vmaf, pic, pic_device, hw_flags);
    case VMAF_PICTURE_BUFFER_TYPE_CUDA_DEVICE:
        *pic_device = *pic;
        return translate_picture_device(vmaf, pic, pic_host, hw_flags);
    default:
        return -EINVAL;
    }
}

static unsigned rfe_hw_flags(RegisteredFeatureExtractors *rfe)
{
    if (!rfe) return -EINVAL;

    unsigned flags = 0;
    for (unsigned i = 0; i < rfe->cnt; i++) {
        flags |= rfe->fex_ctx[i]->fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA ?
            HW_FLAG_DEVICE : HW_FLAG_HOST;
    }

    return flags;
}

#endif

/* Upstream dispatch function. Refactoring is tracked in .workingdir2/OPEN.md. */
/* NOLINTNEXTLINE(readability-function-size) */
int vmaf_read_pictures(VmafContext *vmaf, VmafPicture *ref, VmafPicture *dist,
                       unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (vmaf->flushed) return -EINVAL;
    if (!ref != !dist) return -EINVAL;
    if (!ref && !dist) return flush_context(vmaf);

    int err = 0;

    vmaf->pic_cnt++;
    err = validate_pic_params(vmaf, ref, dist);
    if (err) return err;

#ifdef VMAF_PICTURE_POOL
    err = check_picture_pool(vmaf);
    if (err) return err;
#endif

#ifdef HAVE_CUDA
    err = check_ring_buffer(vmaf);
    if (err) return err;

    const unsigned hw_flags =
        rfe_hw_flags(&vmaf->registered_feature_extractors);

    VmafPicture ref_host = { 0 }, ref_device = { 0 };
    err = translate_picture(vmaf, ref, &ref_host, &ref_device, hw_flags);
    if (err) return err;

    VmafPicture dist_host = { 0 }, dist_device = { 0 };
    err = translate_picture(vmaf, dist, &dist_host, &dist_device, hw_flags);
#endif

#ifdef HAVE_SYCL
    // SYCL upload must happen BEFORE the extractor loop because SYCL
    // queues are in-order — kernels enqueued during submit read shared
    // buffers immediately, unlike the deferred batched model.
    // Auto-initialize shared buffers on the first frame.
    if (vmaf->sycl.state) {
        // Double-buffered: compute reads buf[cur_compute] while upload
        // writes buf[cur_upload].  No need to wait for previous compute
        // before uploading — the buffers are disjoint.  Just wait for the
        // previous upload to finish (in-order copy_queue guarantees this
        // implicitly) before overwriting the upload slot.
        // On the very first frame, skip the wait since nothing is pending.
        if (!vmaf_sycl_get_shared_ref(vmaf->sycl.state)) {
            err = vmaf_sycl_shared_frame_init(vmaf->sycl.state,
                                               ref->w[0], ref->h[0],
                                               ref->bpc);
            if (err) return err;
        }
        err = vmaf_sycl_shared_frame_upload(vmaf->sycl.state, ref, dist);
        if (err) return err;
        // DMA runs asynchronously on copy_queue while the CPU collects
        // previous-frame results below.  vmaf_sycl_graph_submit() will
        // wait for copy_queue just before replaying the command graph.
    }
#endif

    // GPU extractor loop: inline collect-then-submit per extractor.
    // This allows the GPU to start on extractor N's new work while the CPU
    // collects results from extractor N+1's previous frame, overlapping
    // GPU compute with CPU score-writing. Critical for small GPUs (RTX 3050).
    for (unsigned i = 0; i < vmaf->registered_feature_extractors.cnt; i++) {
        VmafFeatureExtractorContext *fex_ctx =
            vmaf->registered_feature_extractors.fex_ctx[i];

        if (!(fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL)) {
            if ((vmaf->cfg.n_subsample > 1) && (index % vmaf->cfg.n_subsample))
                continue;
        }

        if (!(fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA) &&
            !(fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_SYCL) &&
            vmaf->thread_pool) {
#ifdef VMAF_BATCH_THREADING
            if (!(fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL))
#endif
            continue;
        }
#ifdef HAVE_CUDA
        ref = fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA ?
            &ref_device : &ref_host;
        dist = fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA ?
            &dist_device : &dist_host;
#endif

        if ((fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_PREV_REF) &&
            vmaf->prev_ref.ref)
        {
            fex_ctx->fex->prev_ref = vmaf->prev_ref;
        }

        // Double-buffer: collect previous frame, then submit current frame.
        // This overlaps GPU compute of the previous frame with CPU-side
        // command recording for the current frame.
        // Works for both Vulkan and CUDA extractors that implement submit/collect.
        if (fex_ctx->fex->submit && fex_ctx->fex->collect) {
            // Collect previous frame's results (if any)
            if (fex_ctx->gpu_pending) {
                err = vmaf_feature_extractor_context_collect(fex_ctx,
                        fex_ctx->gpu_pending_index,
                        vmaf->feature_collector);
                fex_ctx->gpu_pending = false;
                if (err)
                    return err;
            }
            // Submit current frame (non-blocking GPU work)
            err = vmaf_feature_extractor_context_submit(fex_ctx,
                    ref, NULL, dist, NULL, index);
            if (err)
                return err;
            fex_ctx->gpu_pending = true;
            fex_ctx->gpu_pending_index = index;
            continue;
        }

        err = vmaf_feature_extractor_context_extract(fex_ctx, ref, NULL, dist,
                                                     NULL, index,
                                                     vmaf->feature_collector);

        if (fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_PREV_REF)
            memset(&fex_ctx->fex->prev_ref, 0, sizeof(fex_ctx->fex->prev_ref));

        if (err)
            return err;
    }

    // Note: SYCL upload is done BEFORE the extractor loop (above)
    // because SYCL in-order queues require data to be present when
    // kernels are enqueued.

#ifdef HAVE_CUDA
        ref = &ref_host;
        dist = &dist_host;
#endif

    /* Per-frame tiny-model inference, if one is attached. Runs on the main
     * thread after extractor dispatch; publishes to the feature collector
     * under the sidecar-derived feature name. */
    if (vmaf->dnn.sess) {
        err = vmaf_ctx_dnn_run_frame(vmaf, ref, index);
        if (err) return err;
    }

    //multithreading for GPU does not yield performance benefits
    //disabled for now
    if (vmaf->thread_pool){
#ifdef VMAF_BATCH_THREADING
        return threaded_read_pictures_batch(vmaf, ref, dist, index);
#else
        return threaded_read_pictures(vmaf, ref, dist, index);
#endif
    }

    if (vmaf->prev_ref.ref)
        vmaf_picture_unref(&vmaf->prev_ref);
    vmaf_picture_ref(&vmaf->prev_ref, ref);
#ifdef HAVE_CUDA
    if (ref_host.priv)
        err |= vmaf_picture_unref(&ref_host);

    if (dist_host.priv)
        err |= vmaf_picture_unref(&dist_host);

    CudaFunctions* cu_f = vmaf->cuda.state.f;
    if (ref_device.priv) {
        CHECK_CUDA(cu_f, cuEventRecord(vmaf_cuda_picture_get_finished_event(&ref_device),
                                 vmaf_cuda_picture_get_stream(&ref_device)));
        //^FIXME: move to picture callback
        err |= vmaf_picture_unref(&ref_device);
    }

    if (dist_device.priv) {
        CHECK_CUDA(cu_f, cuEventRecord(vmaf_cuda_picture_get_finished_event(&dist_device),
                                vmaf_cuda_picture_get_stream(&dist_device)));
        //^FIXME: move to picture callback
        err |= vmaf_picture_unref(&dist_device);
    }

#else
    err |= vmaf_picture_unref(ref);
    err |= vmaf_picture_unref(dist);
#endif

    return err;
}

#ifdef HAVE_SYCL
int vmaf_read_pictures_sycl(VmafContext *vmaf, unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!vmaf->sycl.state) return -EINVAL;
    if (vmaf->flushed) return -EINVAL;

    int err = 0;
    vmaf->pic_cnt++;

    // Ensure de-tile kernels on the primary queue have finished reading from
    // imported VA surface memory.  After this function returns, the caller
    // (FFmpeg filter) may release the AVFrame, letting the QSV hwupload pool
    // reuse the VA surface for the next frame.  Without this wait, the async
    // de-tile could race with the hwupload writing new data.
    err = vmaf_sycl_queue_wait(vmaf->sycl.state);
    if (err) return err;

    // Advance frame counter for the zero-copy VA import path.
    // The host upload path (vmaf_sycl_shared_frame_upload) does this
    // internally, but the VA import path bypasses upload entirely.
    // Without this, graph_submit/graph_wait synchronisation breaks:
    //   - graph_wait idempotency causes stale results from frame 2+
    //   - graph_submit fires 3x per frame instead of once
    vmaf_sycl_advance_frame(vmaf->sycl.state);

    // GPU extractor loop: collect previous results, then submit new work.
    // No upload needed — caller already wrote Y plane data into the shared
    // SYCL USM device buffers (e.g. via VPL Level Zero interop).
    for (unsigned i = 0; i < vmaf->registered_feature_extractors.cnt; i++) {
        VmafFeatureExtractorContext *fex_ctx =
            vmaf->registered_feature_extractors.fex_ctx[i];

        if (!(fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_SYCL))
            continue;

        if (!(fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL)) {
            if ((vmaf->cfg.n_subsample > 1) && (index % vmaf->cfg.n_subsample))
                continue;
        }

        // Lazy initialization
        if (!fex_ctx->is_initialized) {
            err = vmaf_feature_extractor_context_init(fex_ctx,
                    vmaf->pic_params.pix_fmt, vmaf->pic_params.bpc,
                    vmaf->pic_params.w, vmaf->pic_params.h);
            if (err) return err;
        }

        // Collect previous frame's results (double-buffered)
        if (fex_ctx->gpu_pending) {
            err = vmaf_feature_extractor_context_collect(fex_ctx,
                    fex_ctx->gpu_pending_index,
                    vmaf->feature_collector);
            fex_ctx->gpu_pending = false;
            if (err) return err;
        }

        // Submit current frame (GPU buffers already populated)
        err = vmaf_feature_extractor_context_submit_nocopy(fex_ctx, index);
        if (err) return err;
        fex_ctx->gpu_pending = true;
        fex_ctx->gpu_pending_index = index;
    }

    return err;
}

int vmaf_flush_sycl(VmafContext *vmaf)
{
    if (!vmaf) return -EINVAL;
    if (vmaf->flushed) return -EINVAL;

    int err = 0;

    if (vmaf->sycl.state) {
        RegisteredFeatureExtractors rfe = vmaf->registered_feature_extractors;
        // Collect any pending double-buffered SYCL work
        for (unsigned i = 0; i < rfe.cnt; i++) {
            if ((rfe.fex_ctx[i]->fex->flags & VMAF_FEATURE_EXTRACTOR_SYCL) &&
                rfe.fex_ctx[i]->gpu_pending) {
                err |= vmaf_feature_extractor_context_collect(rfe.fex_ctx[i],
                        rfe.fex_ctx[i]->gpu_pending_index,
                        vmaf->feature_collector);
                rfe.fex_ctx[i]->gpu_pending = false;
            }
        }
        for (unsigned i = 0; i < rfe.cnt; i++) {
            if (rfe.fex_ctx[i]->fex->flags & VMAF_FEATURE_EXTRACTOR_SYCL)
                err |= vmaf_feature_extractor_context_flush(rfe.fex_ctx[i],
                                                            vmaf->feature_collector);
        }
        vmaf_sycl_queue_wait(vmaf->sycl.state);
        vmaf_sycl_print_timing(vmaf->sycl.state);
    }

    if (!err) vmaf->flushed = true;
    return err;
}
#endif

int vmaf_register_metadata_handler(VmafContext *vmaf, VmafMetadataConfiguration cfg)
{
    if (!vmaf) return -EINVAL;

    return vmaf_feature_collector_register_metadata(vmaf->feature_collector, cfg);
}

int vmaf_feature_score_at_index(VmafContext *vmaf, const char *feature_name,
                                double *score, unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;
    if (!score) return -EINVAL;

    return vmaf_feature_collector_get_score(vmaf->feature_collector,
                                            feature_name, score, index);
}


int vmaf_score_at_index(VmafContext *vmaf, VmafModel *model, double *score,
                        unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!model) return -EINVAL;
    if (!score) return -EINVAL;

    int err =
        vmaf_feature_collector_get_score(vmaf->feature_collector, model->name,
                                         score, index);
    if (err) {
        err = vmaf_predict_score_at_index(model, vmaf->feature_collector, index,
                                          score, true, false, 0);
    }

    return err;
}

int vmaf_score_at_index_model_collection(VmafContext *vmaf,
                                         VmafModelCollection *model_collection,
                                         VmafModelCollectionScore *score,
                                         unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!model_collection) return -EINVAL;
    if (!score) return -EINVAL;

    return vmaf_predict_score_at_index_model_collection(model_collection,
                                                        vmaf->feature_collector,
                                                        index, score);
}

int vmaf_feature_score_pooled(VmafContext *vmaf, const char *feature_name,
                              enum VmafPoolingMethod pool_method, double *score,
                              unsigned index_low, unsigned index_high)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;
    if (index_low > index_high) return -EINVAL;
    if (!pool_method) return -EINVAL;

    unsigned pic_cnt = 0;
    double min = 0.;
    double max = 0.;
    double sum = 0.;
    double i_sum = 0.;
    for (unsigned i = index_low; i <= index_high; i++) {
        if ((vmaf->cfg.n_subsample > 1) && (i % vmaf->cfg.n_subsample))
            continue;
        pic_cnt++;
        double s;
        int err = vmaf_feature_score_at_index(vmaf, feature_name, &s, i);
        if (err) return err;
        sum += s;
        i_sum += 1. / (s + 1.);
        if ((i == index_low) || (s < min))
            min = s;
        if ((i == index_low) || (s > max))
            max = s;
    }

    switch (pool_method) {
    case VMAF_POOL_METHOD_MEAN:
        *score = sum / pic_cnt;
        break;
    case VMAF_POOL_METHOD_MIN:
        *score = min;
        break;
    case VMAF_POOL_METHOD_MAX:
        *score = max;
        break;
    case VMAF_POOL_METHOD_HARMONIC_MEAN:
        *score = pic_cnt / i_sum - 1.0;
        break;
    default:
        return -EINVAL;
    }

    return 0;
}

int vmaf_score_pooled(VmafContext *vmaf, VmafModel *model,
                      enum VmafPoolingMethod pool_method, double *score,
                      unsigned index_low, unsigned index_high)
{
    if (!vmaf) return -EINVAL;
    if (!model) return -EINVAL;
    if (!score) return -EINVAL;
    if (index_low > index_high) return -EINVAL;
    if (!pool_method) return -EINVAL;

    for (unsigned i = index_low; i <= index_high; i++) {
        if ((vmaf->cfg.n_subsample > 1) && (i % vmaf->cfg.n_subsample))
            continue;
        double vmaf_score;
        int err = vmaf_score_at_index(vmaf, model, &vmaf_score, i);
        if (err) return err;
    }

    return vmaf_feature_score_pooled(vmaf, model->name, pool_method, score,
                                     index_low, index_high);
}

int vmaf_score_pooled_model_collection(VmafContext *vmaf,
                                       VmafModelCollection *model_collection,
                                       enum VmafPoolingMethod pool_method,
                                       VmafModelCollectionScore *score,
                                       unsigned index_low, unsigned index_high)
{
    if (!vmaf) return -EINVAL;
    if (!model_collection) return -EINVAL;
    if (!score) return -EINVAL;
    if (index_low > index_high) return -EINVAL;
    if (!pool_method) return -EINVAL;

    int err = 0;
    for (unsigned i = index_low; i <= index_high; i++) {
        if ((vmaf->cfg.n_subsample > 1) && (i % vmaf->cfg.n_subsample))
            continue;
        VmafModelCollectionScore s;
        err = vmaf_score_at_index_model_collection(vmaf, model_collection, &s, i);
        if (err) return err;
    }

    score->type = VMAF_MODEL_COLLECTION_SCORE_BOOTSTRAP;

    //TODO: dedupe, vmaf_bootstrap_predict_score_at_index()
    const char *suffix_lo = "_ci_p95_lo";
    const char *suffix_hi = "_ci_p95_hi";
    const char *suffix_bagging = "_bagging";
    const char *suffix_stddev = "_stddev";
    const size_t name_sz =
        strlen(model_collection->name) + strlen(suffix_lo) + 1;
    char name[name_sz];
    memset(name, 0, name_sz);

    (void) snprintf(name, name_sz, "%s%s", model_collection->name, suffix_bagging);
    err |= vmaf_feature_score_pooled(vmaf, name, pool_method,
                                     &score->bootstrap.bagging_score,
                                     index_low, index_high);

    (void) snprintf(name, name_sz, "%s%s", model_collection->name, suffix_stddev);
    err |= vmaf_feature_score_pooled(vmaf, name, pool_method,
                                     &score->bootstrap.stddev,
                                     index_low, index_high);

    (void) snprintf(name, name_sz, "%s%s", model_collection->name, suffix_lo);
    err |= vmaf_feature_score_pooled(vmaf, name, pool_method,
                                     &score->bootstrap.ci.p95.lo,
                                     index_low, index_high);

    (void) snprintf(name, name_sz, "%s%s", model_collection->name, suffix_hi);
    err |= vmaf_feature_score_pooled(vmaf, name, pool_method,
                                     &score->bootstrap.ci.p95.hi,
                                     index_low, index_high);

    return err;
}

const char *vmaf_version(void)
{
    return VMAF_VERSION;
}

int vmaf_write_output_with_format(VmafContext *vmaf, const char *output_path,
                                  enum VmafOutputFormat fmt,
                                  const char *score_format)
{
    FILE *outfile = fopen(output_path, "w");
    if (!outfile) {
        (void) fprintf(stderr, "could not open file: %s\n", output_path);
        return -EINVAL;
    }

    const double fps = vmaf->pic_cnt /
                ((double) (vmaf->feature_collector->timer.end -
                vmaf->feature_collector->timer.begin) / CLOCKS_PER_SEC);

    int ret = 0;
    switch (fmt) {
    case VMAF_OUTPUT_FORMAT_XML:
        ret = vmaf_write_output_xml(vmaf, vmaf->feature_collector, outfile,
                                    vmaf->cfg.n_subsample,
                                    vmaf->pic_params.w, vmaf->pic_params.h,
                                    fps, vmaf->pic_cnt, score_format);
        break;
    case VMAF_OUTPUT_FORMAT_JSON:
        ret = vmaf_write_output_json(vmaf, vmaf->feature_collector, outfile,
                                     vmaf->cfg.n_subsample, fps, vmaf->pic_cnt,
                                     score_format);
        break;
    case VMAF_OUTPUT_FORMAT_CSV:
        ret = vmaf_write_output_csv(vmaf->feature_collector, outfile,
                                    vmaf->cfg.n_subsample, score_format);
        break;
    case VMAF_OUTPUT_FORMAT_SUB:
        ret = vmaf_write_output_sub(vmaf->feature_collector, outfile,
                                    vmaf->cfg.n_subsample, score_format);
        break;
    default:
        ret = -EINVAL;
        break;
    }

    if (fclose(outfile) != 0 && ret == 0) ret = -EIO;
    return ret;
}

int vmaf_write_output(VmafContext *vmaf, const char *output_path,
                      enum VmafOutputFormat fmt)
{
    return vmaf_write_output_with_format(vmaf, output_path, fmt, NULL);
}
