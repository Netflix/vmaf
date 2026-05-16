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

#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#include "config.h"
#include "feature_extractor.h"
#include "log.h"
#include "picture.h"

#ifdef HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#if VMAF_FLOAT_FEATURES
extern VmafFeatureExtractor vmaf_fex_float_psnr;
extern VmafFeatureExtractor vmaf_fex_float_ansnr;
extern VmafFeatureExtractor vmaf_fex_float_adm;
extern VmafFeatureExtractor vmaf_fex_float_motion;
extern VmafFeatureExtractor vmaf_fex_float_moment;
extern VmafFeatureExtractor vmaf_fex_float_vif;
extern VmafFeatureExtractor vmaf_fex_speed_chroma;
extern VmafFeatureExtractor vmaf_fex_speed_temporal;
#endif
extern VmafFeatureExtractor vmaf_fex_float_ssim;
extern VmafFeatureExtractor vmaf_fex_float_ms_ssim;
extern VmafFeatureExtractor vmaf_fex_ssim;
extern VmafFeatureExtractor vmaf_fex_ssimulacra2;
extern VmafFeatureExtractor vmaf_fex_ciede;
extern VmafFeatureExtractor vmaf_fex_psnr;
extern VmafFeatureExtractor vmaf_fex_psnr_hvs;
extern VmafFeatureExtractor vmaf_fex_integer_adm;
extern VmafFeatureExtractor vmaf_fex_integer_motion;
extern VmafFeatureExtractor vmaf_fex_integer_motion_v2;
extern VmafFeatureExtractor vmaf_fex_integer_vif;
extern VmafFeatureExtractor vmaf_fex_cambi;
#if HAVE_CUDA
extern VmafFeatureExtractor vmaf_fex_integer_adm_cuda;
extern VmafFeatureExtractor vmaf_fex_integer_vif_cuda;
extern VmafFeatureExtractor vmaf_fex_integer_motion_cuda;
extern VmafFeatureExtractor vmaf_fex_integer_motion_v2_cuda;
extern VmafFeatureExtractor vmaf_fex_psnr_cuda;
extern VmafFeatureExtractor vmaf_fex_float_moment_cuda;
extern VmafFeatureExtractor vmaf_fex_ciede_cuda;
extern VmafFeatureExtractor vmaf_fex_float_ssim_cuda;
extern VmafFeatureExtractor vmaf_fex_float_ms_ssim_cuda;
extern VmafFeatureExtractor vmaf_fex_psnr_hvs_cuda;
extern VmafFeatureExtractor vmaf_fex_float_ansnr_cuda;
extern VmafFeatureExtractor vmaf_fex_float_psnr_cuda;
extern VmafFeatureExtractor vmaf_fex_float_motion_cuda;
extern VmafFeatureExtractor vmaf_fex_float_vif_cuda;
extern VmafFeatureExtractor vmaf_fex_ssimulacra2_cuda;
extern VmafFeatureExtractor vmaf_fex_float_adm_cuda;
/* T3-15 / ADR-0360: cambi CUDA twin (Strategy II hybrid). */
extern VmafFeatureExtractor vmaf_fex_cambi_cuda;
#endif
#if HAVE_SYCL
extern VmafFeatureExtractor vmaf_fex_integer_vif_sycl;
extern VmafFeatureExtractor vmaf_fex_integer_adm_sycl;
extern VmafFeatureExtractor vmaf_fex_integer_motion_sycl;
extern VmafFeatureExtractor vmaf_fex_integer_motion_v2_sycl;
extern VmafFeatureExtractor vmaf_fex_psnr_sycl;
extern VmafFeatureExtractor vmaf_fex_float_moment_sycl;
extern VmafFeatureExtractor vmaf_fex_ciede_sycl;
extern VmafFeatureExtractor vmaf_fex_float_ssim_sycl;
extern VmafFeatureExtractor vmaf_fex_float_ms_ssim_sycl;
extern VmafFeatureExtractor vmaf_fex_psnr_hvs_sycl;
extern VmafFeatureExtractor vmaf_fex_float_ansnr_sycl;
extern VmafFeatureExtractor vmaf_fex_float_psnr_sycl;
extern VmafFeatureExtractor vmaf_fex_float_motion_sycl;
extern VmafFeatureExtractor vmaf_fex_float_vif_sycl;
extern VmafFeatureExtractor vmaf_fex_ssimulacra2_sycl;
extern VmafFeatureExtractor vmaf_fex_float_adm_sycl;
/* T3-15 / ADR-0371: cambi SYCL twin (Strategy II hybrid, closes CUDA→SYCL parity gap). */
extern VmafFeatureExtractor vmaf_fex_cambi_sycl;
#endif
#if HAVE_VULKAN
extern VmafFeatureExtractor vmaf_fex_integer_vif_vulkan;
extern VmafFeatureExtractor vmaf_fex_integer_motion_vulkan;
extern VmafFeatureExtractor vmaf_fex_integer_motion_v2_vulkan;
extern VmafFeatureExtractor vmaf_fex_integer_adm_vulkan;
extern VmafFeatureExtractor vmaf_fex_psnr_vulkan;
extern VmafFeatureExtractor vmaf_fex_float_moment_vulkan;
extern VmafFeatureExtractor vmaf_fex_ciede_vulkan;
extern VmafFeatureExtractor vmaf_fex_float_ssim_vulkan;
extern VmafFeatureExtractor vmaf_fex_float_ms_ssim_vulkan;
extern VmafFeatureExtractor vmaf_fex_psnr_hvs_vulkan;
extern VmafFeatureExtractor vmaf_fex_float_ansnr_vulkan;
extern VmafFeatureExtractor vmaf_fex_float_psnr_vulkan;
extern VmafFeatureExtractor vmaf_fex_float_motion_vulkan;
extern VmafFeatureExtractor vmaf_fex_float_vif_vulkan;
extern VmafFeatureExtractor vmaf_fex_float_adm_vulkan;
extern VmafFeatureExtractor vmaf_fex_ssimulacra2_vulkan;
extern VmafFeatureExtractor vmaf_fex_cambi_vulkan;
#endif
#if HAVE_HIP
/* HIP first-consumer kernel — T7-10 / ADR-0241. Registration succeeds
 * but `init()` returns -ENOSYS until the runtime PR (T7-10b) replaces
 * the kernel-template helper bodies with real HIP calls. */
extern VmafFeatureExtractor vmaf_fex_psnr_hip;
/* HIP second-consumer kernel — T7-10b / ADR-0254. First real kernel:
 * float_psnr_hip. With `enable_hipcc=true` the HSACO is embedded and
 * the kernel runs on device; without it init() returns -ENOSYS. */
extern VmafFeatureExtractor vmaf_fex_float_psnr_hip;
/* HIP third-consumer kernel — T7-10b follow-up / ADR-0257. Same
 * scaffold posture as the first consumer: registration succeeds,
 * `init()` returns -ENOSYS until T7-10b. Mirrors
 * `vmaf_fex_ciede_cuda` field-for-field. */
extern VmafFeatureExtractor vmaf_fex_ciede_hip;
/* HIP fourth-consumer kernel — T7-10b follow-up / ADR-0258. Same
 * scaffold posture; emits four `float_moment_*` features. */
extern VmafFeatureExtractor vmaf_fex_float_moment_hip;
/* HIP fifth/sixth consumers — ADR-0266 / ADR-0267. Same posture as
 * the first consumer: registration succeeds, `init()` returns
 * -ENOSYS until T7-10b. */
extern VmafFeatureExtractor vmaf_fex_float_ansnr_hip;
extern VmafFeatureExtractor vmaf_fex_integer_motion_v2_hip;
/* HIP seventh-consumer kernel — T7-10b follow-up / ADR-0273. Same
 * scaffold posture; mirrors the CUDA twin
 * `feature/cuda/float_motion_cuda.c` and pins the temporal-extractor
 * shape with a raw-pixel cache + blurred-frame ping-pong slot pair. */
extern VmafFeatureExtractor vmaf_fex_float_motion_hip;
/* HIP eighth-consumer kernel — T7-10b follow-up / ADR-0274. Same
 * scaffold posture; mirrors the CUDA twin
 * `feature/cuda/integer_ssim_cuda.c` and pins the two-dispatch +
 * five intermediate float buffers shape. v1: scale=1 only. */
extern VmafFeatureExtractor vmaf_fex_float_ssim_hip;
#endif
#if HAVE_METAL
/* Metal feature extractors — T8-1c through T8-1j / ADR-0421.
 * All eight consumers are fully implemented as Obj-C++ .mm dispatch
 * files in feature/metal/; the -ENOSYS scaffold .c stubs have been
 * removed. Kernels live under feature/metal/, compiled via xcrun
 * into the __TEXT,__metallib section of libvmaf. */
extern VmafFeatureExtractor vmaf_fex_integer_motion_v2_metal;
extern VmafFeatureExtractor vmaf_fex_integer_psnr_metal;
extern VmafFeatureExtractor vmaf_fex_float_ssim_metal;
extern VmafFeatureExtractor vmaf_fex_integer_motion_metal;
extern VmafFeatureExtractor vmaf_fex_float_psnr_metal;
extern VmafFeatureExtractor vmaf_fex_float_ansnr_metal;
extern VmafFeatureExtractor vmaf_fex_float_motion_metal;
extern VmafFeatureExtractor vmaf_fex_float_moment_metal;
#endif
/* SpEED-QA NR metric scaffold — ADR-0253. */
extern VmafFeatureExtractor vmaf_fex_speed_qa;
extern VmafFeatureExtractor vmaf_fex_lpips;
extern VmafFeatureExtractor vmaf_fex_dists_sq;
extern VmafFeatureExtractor vmaf_fex_fastdvdnet_pre;
extern VmafFeatureExtractor vmaf_fex_mobilesal;
extern VmafFeatureExtractor vmaf_fex_transnet_v2;
extern VmafFeatureExtractor vmaf_fex_null;

static VmafFeatureExtractor *feature_extractor_list[] = {
#if VMAF_FLOAT_FEATURES
    &vmaf_fex_float_psnr, &vmaf_fex_float_ansnr, &vmaf_fex_float_adm, &vmaf_fex_float_vif,
    &vmaf_fex_float_motion, &vmaf_fex_float_moment, &vmaf_fex_speed_chroma,
    &vmaf_fex_speed_temporal,
#endif
    &vmaf_fex_float_ms_ssim, &vmaf_fex_float_ssim, &vmaf_fex_ssim, &vmaf_fex_ssimulacra2,
    &vmaf_fex_ciede, &vmaf_fex_psnr, &vmaf_fex_psnr_hvs, &vmaf_fex_integer_adm,
    &vmaf_fex_integer_motion, &vmaf_fex_integer_motion_v2, &vmaf_fex_integer_vif, &vmaf_fex_cambi,
#if HAVE_SYCL
    // SYCL before CUDA: when multiple GPU backends are compiled in,
    // the first matching extractor wins.  SYCL is the preferred backend
    // because it supports the widest range of Intel hardware.
    // Use --no_sycl to fall back to CUDA.
    &vmaf_fex_integer_vif_sycl, &vmaf_fex_integer_adm_sycl, &vmaf_fex_integer_motion_sycl,
    &vmaf_fex_integer_motion_v2_sycl, &vmaf_fex_psnr_sycl, &vmaf_fex_float_moment_sycl,
    &vmaf_fex_ciede_sycl, &vmaf_fex_float_ssim_sycl, &vmaf_fex_float_ms_ssim_sycl,
    &vmaf_fex_psnr_hvs_sycl, &vmaf_fex_psnr_sycl, &vmaf_fex_float_moment_sycl, &vmaf_fex_ciede_sycl,
    &vmaf_fex_float_ssim_sycl, &vmaf_fex_float_ms_ssim_sycl, &vmaf_fex_psnr_hvs_sycl,
    &vmaf_fex_float_ansnr_sycl, &vmaf_fex_float_psnr_sycl, &vmaf_fex_float_motion_sycl,
    &vmaf_fex_float_vif_sycl, &vmaf_fex_ssimulacra2_sycl, &vmaf_fex_float_adm_sycl,
    /* T3-15 / ADR-0371: cambi SYCL twin (closes last CUDA→SYCL parity gap). */
    &vmaf_fex_cambi_sycl,
#endif
#if HAVE_VULKAN
    /* Vulkan is registered AFTER SYCL/CUDA — those backends remain
     * the preferred GPU paths because they ship the full feature
     * set (ADM + motion + VIF) and are bit-exact-tested against the
     * Netflix golden gate. Vulkan is opt-in via explicit feature
     * name selection until the full backend lands (T5-1c). */
    &vmaf_fex_integer_vif_vulkan, &vmaf_fex_integer_motion_vulkan,
    &vmaf_fex_integer_motion_v2_vulkan, &vmaf_fex_integer_adm_vulkan, &vmaf_fex_psnr_vulkan,
    &vmaf_fex_float_moment_vulkan, &vmaf_fex_ciede_vulkan, &vmaf_fex_float_ssim_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_integer_vif_vulkan,
    &vmaf_fex_integer_motion_vulkan, &vmaf_fex_integer_adm_vulkan, &vmaf_fex_psnr_vulkan,
    &vmaf_fex_float_moment_vulkan, &vmaf_fex_ciede_vulkan, &vmaf_fex_float_ssim_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_float_ansnr_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_integer_vif_vulkan,
    &vmaf_fex_integer_motion_vulkan, &vmaf_fex_integer_adm_vulkan, &vmaf_fex_psnr_vulkan,
    &vmaf_fex_float_moment_vulkan, &vmaf_fex_ciede_vulkan, &vmaf_fex_float_ssim_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_float_psnr_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_integer_vif_vulkan,
    &vmaf_fex_integer_motion_vulkan, &vmaf_fex_integer_adm_vulkan, &vmaf_fex_psnr_vulkan,
    &vmaf_fex_float_moment_vulkan, &vmaf_fex_ciede_vulkan, &vmaf_fex_float_ssim_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_float_motion_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_integer_vif_vulkan,
    &vmaf_fex_integer_motion_vulkan, &vmaf_fex_integer_adm_vulkan, &vmaf_fex_psnr_vulkan,
    &vmaf_fex_float_moment_vulkan, &vmaf_fex_ciede_vulkan, &vmaf_fex_float_ssim_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_float_vif_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_integer_vif_vulkan,
    &vmaf_fex_integer_motion_vulkan, &vmaf_fex_integer_adm_vulkan, &vmaf_fex_psnr_vulkan,
    &vmaf_fex_float_moment_vulkan, &vmaf_fex_ciede_vulkan, &vmaf_fex_float_ssim_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_float_adm_vulkan,
    &vmaf_fex_float_ms_ssim_vulkan, &vmaf_fex_psnr_hvs_vulkan, &vmaf_fex_ssimulacra2_vulkan,
    /* T7-36 / ADR-0205: cambi Vulkan twin (Strategy II hybrid). */
    &vmaf_fex_cambi_vulkan,
#endif
#if HAVE_CUDA
    &vmaf_fex_integer_adm_cuda, &vmaf_fex_integer_vif_cuda, &vmaf_fex_integer_motion_cuda,
    &vmaf_fex_integer_motion_v2_cuda, &vmaf_fex_psnr_cuda, &vmaf_fex_float_moment_cuda,
    &vmaf_fex_ciede_cuda, &vmaf_fex_float_ssim_cuda, &vmaf_fex_float_ms_ssim_cuda,
    &vmaf_fex_psnr_hvs_cuda, &vmaf_fex_float_ansnr_cuda, &vmaf_fex_float_psnr_cuda,
    &vmaf_fex_float_motion_cuda, &vmaf_fex_float_vif_cuda, &vmaf_fex_ssimulacra2_cuda,
    &vmaf_fex_float_adm_cuda,
    /* T3-15 / ADR-0360: cambi CUDA twin (Strategy II hybrid). */
    &vmaf_fex_cambi_cuda,
#endif
#if HAVE_HIP
    /* T7-10 first consumer (ADR-0241): registration succeeds even on
     * the scaffold-only build so a caller asking for `psnr_hip` gets
     * the cleaner "extractor found, runtime not ready (-ENOSYS)"
     * surface instead of "no such extractor". The runtime PR
     * (T7-10b) keeps this row verbatim and adds its siblings. */
    &vmaf_fex_psnr_hip,
    /* T7-10b second consumer (ADR-0254): first real kernel. With
     * `enable_hipcc=true` the HSACO is loaded and the kernel runs
     * on device; without it init() returns -ENOSYS (scaffold posture).
     * Emits `float_psnr` (luma-only, same as the CUDA twin). */
    &vmaf_fex_float_psnr_hip,
    /* Third consumer (ADR-0257): `ciede_hip` mirrors
     * `integer_ciede_cuda.c`'s call graph. Same scaffold-only
     * registration posture — registers, `init()` returns -ENOSYS
     * until T7-10b. */
    &vmaf_fex_ciede_hip,
    /* Fourth consumer (ADR-0258): `float_moment_hip` mirrors
     * `integer_moment_cuda.c`'s call graph; emits four
     * `float_moment_*` features once the runtime kernel arrives. */
    &vmaf_fex_float_moment_hip,
    /* T7-10b fifth + sixth consumers (ADR-0266 / ADR-0267): same
     * scaffold-posture registration as the first consumer. */
    &vmaf_fex_float_ansnr_hip, &vmaf_fex_integer_motion_v2_hip,
    /* Seventh consumer (ADR-0273): `float_motion_hip` mirrors
     * `float_motion_cuda.c`'s call graph (TEMPORAL flag,
     * raw-pixel cache + blurred-frame ping-pong, `flush()`
     * tail-frame motion2 emission); emits two features
     * (`VMAF_feature_motion_score`, `VMAF_feature_motion2_score`)
     * once the runtime kernel arrives. */
    &vmaf_fex_float_motion_hip,
    /* Eighth consumer (ADR-0274): `float_ssim_hip` mirrors
     * `integer_ssim_cuda.c`'s call graph (two-dispatch separable
     * Gaussian, five intermediate float buffers, per-block
     * float-partial readback); emits one feature (`float_ssim`)
     * once the runtime kernel arrives. v1 is scale=1 only. */
    &vmaf_fex_float_ssim_hip,
#endif
#if HAVE_METAL
    /* T8-1 first consumer (ADR-0361): registration succeeds even on
     * the scaffold-only build so a caller asking for `motion_v2_metal`
     * gets the cleaner "extractor found, runtime not ready (-ENOSYS)"
     * surface instead of "no such extractor". The runtime PR (T8-1b /
     * T8-1c) keeps this row verbatim and adds its siblings. */
    &vmaf_fex_integer_motion_v2_metal,
    /* T8-1 batch-1 additional consumers (ADR-0361). */
    &vmaf_fex_integer_psnr_metal, &vmaf_fex_float_ssim_metal, &vmaf_fex_integer_motion_metal,
    /* T8-1 batch-2 additional consumers (ADR-0361): 4 float features. */
    &vmaf_fex_float_psnr_metal, &vmaf_fex_float_ansnr_metal, &vmaf_fex_float_motion_metal,
    &vmaf_fex_float_moment_metal,
#endif
    &vmaf_fex_speed_qa, &vmaf_fex_lpips, &vmaf_fex_dists_sq, &vmaf_fex_fastdvdnet_pre,
    &vmaf_fex_mobilesal, &vmaf_fex_transnet_v2, &vmaf_fex_null, NULL};

VmafFeatureExtractor *vmaf_get_feature_extractor_by_name(const char *name)
{
    if (!name)
        return NULL;

    VmafFeatureExtractor *fex = NULL;
    for (unsigned i = 0; (fex = feature_extractor_list[i]); i++) {
        if (!strcmp(name, fex->name))
            return fex;
    }

    return NULL;
}

VmafFeatureExtractor *vmaf_get_feature_extractor_by_feature_name(const char *name, unsigned flags)
{
    if (!name)
        return NULL;

    VmafFeatureExtractor *fex = NULL;

    for (unsigned i = 0; (fex = feature_extractor_list[i]); i++) {
        if (!fex->provided_features)
            continue;
        if (flags && !(fex->flags & flags))
            continue;
        const char *fname = NULL;
        for (unsigned j = 0; (fname = fex->provided_features[j]); j++) {
            if (!strcmp(name, fname))
                return fex;
        }
    }
    return NULL;
}

static int vmaf_fex_ctx_parse_options(VmafFeatureExtractorContext *fex_ctx)
{
    const VmafOption *opt = NULL;
    for (unsigned i = 0; (opt = &fex_ctx->fex->options[i]); i++) {
        if (!opt->name)
            break;
        const VmafDictionaryEntry *entry = vmaf_dictionary_get(&fex_ctx->opts_dict, opt->name, 0);
        int err = vmaf_option_set(opt, fex_ctx->fex->priv, entry ? entry->val : NULL);
        if (err)
            return -EINVAL;
    }

    return 0;
}

int vmaf_feature_extractor_context_create(VmafFeatureExtractorContext **fex_ctx,
                                          VmafFeatureExtractor *fex, VmafDictionary *opts_dict)
{
    VmafFeatureExtractorContext *f = *fex_ctx = malloc(sizeof(*f));
    if (!f)
        return -ENOMEM;
    memset(f, 0, sizeof(*f));

    VmafFeatureExtractor *x = malloc(sizeof(*x));
    if (!x)
        goto free_f;
    memcpy(x, fex, sizeof(*x));

    f->fex = x;
    if (f->fex->priv_size) {
        void *priv = malloc(f->fex->priv_size);
        if (!priv)
            goto free_x;
        memset(priv, 0, f->fex->priv_size);
        f->fex->priv = priv;
    }

    f->opts_dict = opts_dict;
    if (f->fex->options && f->fex->priv) {
        int err = vmaf_fex_ctx_parse_options(f);
        if (err)
            return err;
    }

    return 0;

free_x:
    free(x);
free_f:
    free(f);
    return -ENOMEM;
}

int vmaf_feature_extractor_context_init(VmafFeatureExtractorContext *fex_ctx,
                                        enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                                        unsigned h)
{
    if (!fex_ctx)
        return -EINVAL;
    if (fex_ctx->is_initialized)
        return -EINVAL;
    if (!pix_fmt)
        return -EINVAL;

    if (fex_ctx->fex->init && !fex_ctx->is_initialized) {
        int err = fex_ctx->fex->init(fex_ctx->fex, pix_fmt, bpc, w, h);
        if (err)
            return err;
    }

    fex_ctx->is_initialized = true;
    return 0;
}

int vmaf_feature_extractor_context_extract(VmafFeatureExtractorContext *fex_ctx, VmafPicture *ref,
                                           VmafPicture *ref_90, VmafPicture *dist,
                                           VmafPicture *dist_90, unsigned pic_index,
                                           VmafFeatureCollector *vfc)
{
    if (!fex_ctx)
        return -EINVAL;
    if (!ref)
        return -EINVAL;
    if (!dist)
        return -EINVAL;
    if (!vfc)
        return -EINVAL;
    if (!fex_ctx->fex->extract)
        return -EINVAL;

    VmafPicturePrivate *ref_priv = ref->priv;
    if (fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_CUDA) {
        if (ref_priv->buf_type != VMAF_PICTURE_BUFFER_TYPE_CUDA_DEVICE) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "picture buf_type mismatch: cuda fex (%s), cpu buf\n",
                     fex_ctx->fex->name);
            return -EINVAL;
        }
    } else {
        if (ref_priv->buf_type == VMAF_PICTURE_BUFFER_TYPE_CUDA_DEVICE) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "picture buf_type mismatch: cpu fex (%s), cuda buf\n",
                     fex_ctx->fex->name);
            return -EINVAL;
        }
    }

#ifdef HAVE_NVTX
    nvtxRangePushA(fex_ctx->fex->name);
#endif

    if (!fex_ctx->is_initialized) {
        int err = vmaf_feature_extractor_context_init(fex_ctx, ref->pix_fmt, ref->bpc, ref->w[0],
                                                      ref->h[0]);
        if (err)
            return err;
    }

    int err = fex_ctx->fex->extract(fex_ctx->fex, ref, ref_90, dist, dist_90, pic_index, vfc);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING, "problem with feature extractor \"%s\" at index %d\n",
                 fex_ctx->fex->name, pic_index);
    }

#ifdef HAVE_NVTX
    nvtxRangePop();
#endif

    return err;
}

int vmaf_feature_extractor_context_submit(VmafFeatureExtractorContext *fex_ctx, VmafPicture *ref,
                                          VmafPicture *ref_90, VmafPicture *dist,
                                          VmafPicture *dist_90, unsigned pic_index)
{
    if (!fex_ctx)
        return -EINVAL;
    if (!ref)
        return -EINVAL;
    if (!dist)
        return -EINVAL;
    if (!fex_ctx->fex->submit)
        return -EINVAL;

    if (!fex_ctx->is_initialized) {
        int err = vmaf_feature_extractor_context_init(fex_ctx, ref->pix_fmt, ref->bpc, ref->w[0],
                                                      ref->h[0]);
        if (err)
            return err;
    }

    return fex_ctx->fex->submit(fex_ctx->fex, ref, ref_90, dist, dist_90, pic_index);
}

int vmaf_feature_extractor_context_submit_nocopy(VmafFeatureExtractorContext *fex_ctx,
                                                 unsigned pic_index)
{
    if (!fex_ctx)
        return -EINVAL;
    if (!fex_ctx->fex->submit)
        return -EINVAL;
    if (!fex_ctx->is_initialized)
        return -EINVAL;

    return fex_ctx->fex->submit(fex_ctx->fex, NULL, NULL, NULL, NULL, pic_index);
}

int vmaf_feature_extractor_context_collect(VmafFeatureExtractorContext *fex_ctx, unsigned pic_index,
                                           VmafFeatureCollector *vfc)
{
    if (!fex_ctx)
        return -EINVAL;
    if (!vfc)
        return -EINVAL;
    if (!fex_ctx->fex->collect)
        return -EINVAL;

    return fex_ctx->fex->collect(fex_ctx->fex, pic_index, vfc);
}

int vmaf_feature_extractor_context_flush(VmafFeatureExtractorContext *fex_ctx,
                                         VmafFeatureCollector *vfc)
{
    if (!fex_ctx)
        return -EINVAL;
    if (!fex_ctx->is_initialized)
        return -EINVAL;
    if (fex_ctx->is_closed)
        return 0;

    int err = 0;
    if (fex_ctx->fex->flush) {
        while (!(err = fex_ctx->fex->flush(fex_ctx->fex, vfc))) {
            /* drain all flush-emitted features */
        }
    }
    return err < 0 ? err : 0;
}

int vmaf_feature_extractor_context_close(VmafFeatureExtractorContext *fex_ctx)
{
    if (!fex_ctx)
        return -EINVAL;
    if (!fex_ctx->is_initialized)
        return -EINVAL;
    if (fex_ctx->is_closed)
        return 0;

    int err = 0;
    if (fex_ctx->fex->close)
        err = fex_ctx->fex->close(fex_ctx->fex);
    fex_ctx->is_closed = true;
    return err;
}

int vmaf_feature_extractor_context_destroy(VmafFeatureExtractorContext *fex_ctx)
{
    if (!fex_ctx)
        return -EINVAL;

    if (fex_ctx->fex) {
        /* free(NULL) is well-defined per C99 §7.20.3.2 / POSIX free(3);
         * the NULL guard is redundant. CodeQL cpp/guarded-free. */
        free(fex_ctx->fex->priv);
        free(fex_ctx->fex);
    }
    if (fex_ctx->opts_dict)
        vmaf_dictionary_free(&fex_ctx->opts_dict);
    free(fex_ctx);
    return 0;
}

int vmaf_fex_ctx_pool_create(VmafFeatureExtractorContextPool **pool, unsigned n_threads)
{
    if (!pool)
        return -EINVAL;
    if (!n_threads)
        return -EINVAL;

    VmafFeatureExtractorContextPool *const p = *pool = malloc(sizeof(*p));
    if (!p)
        goto fail;
    memset(p, 0, sizeof(*p));

    p->n_threads = n_threads;

    p->cnt = 0;
    p->capacity = 8;
    const size_t fex_list_sz = sizeof(*(p->fex_list)) * p->capacity;
    p->fex_list = malloc(fex_list_sz);
    if (!p->fex_list)
        goto free_p;
    memset(p->fex_list, 0, fex_list_sz);

    pthread_mutex_init(&(p->lock), NULL);
    return 0;

free_p:
    free(p);
fail:
    return -ENOMEM;
}

static struct fex_list_entry *get_fex_list_entry(VmafFeatureExtractorContextPool *pool,
                                                 VmafFeatureExtractor *fex,
                                                 VmafDictionary *opts_dict)
{
    if (!pool)
        return NULL;
    if (!fex)
        return NULL;

    for (unsigned i = 0; i < pool->cnt; i++) {
        struct fex_list_entry *entry = &pool->fex_list[i];
        if (!strcmp(fex->name, entry->fex->name) &&
            !vmaf_dictionary_compare(opts_dict, entry->opts_dict)) {
            return entry;
        }
    }

    struct fex_list_entry entry = {0};

    entry.fex = fex;
    const unsigned n_threads = (fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL ? 1 : pool->n_threads);
    atomic_init(&entry.capacity, n_threads);
    atomic_init(&entry.in_use, 0);
    pthread_cond_init(&(entry.full), NULL);
    size_t ctx_array_sz = sizeof(entry.ctx_list[0]) * entry.capacity;
    entry.ctx_list = malloc(ctx_array_sz);
    if (!entry.ctx_list)
        goto fail;
    memset(entry.ctx_list, 0, ctx_array_sz);
    vmaf_dictionary_copy(&opts_dict, &entry.opts_dict);

    if (pool->cnt >= pool->capacity) {
        assert(pool->capacity > 0);
        const size_t capacity = (size_t)pool->capacity * 2;
        struct fex_list_entry *fex_list =
            realloc(pool->fex_list, sizeof(*(pool->fex_list)) * capacity);
        if (!fex_list)
            goto free_ctx_list;
        pool->fex_list = fex_list;
        pool->capacity = capacity;
    }

    pool->fex_list[pool->cnt] = entry;
    return &pool->fex_list[pool->cnt++];

free_ctx_list:
    free(entry.ctx_list);
fail:
    return NULL;
}

static int ctx_pool_ensure_slot_ctx(struct fex_list_entry *entry, int i, VmafFeatureExtractor *fex,
                                    VmafDictionary *opts_dict)
{
    if (entry->ctx_list[i].fex_ctx)
        return 0;

    VmafDictionary *d = NULL;
    if (opts_dict) {
        int err = vmaf_dictionary_copy(&opts_dict, &d);
        if (err) {
            (void)vmaf_dictionary_free(&d);
            return err;
        }
    }
    VmafFeatureExtractorContext *f = NULL;
    int err = vmaf_feature_extractor_context_create(&f, entry->fex, d);
    if (err) {
        (void)vmaf_dictionary_free(&d);
        return err;
    }
    entry->ctx_list[i].fex_ctx = f;
    if (f->fex->flags & VMAF_FEATURE_FRAME_SYNC) {
        f->fex->framesync = (fex->framesync);
    }
    return 0;
}

static int ctx_pool_claim_slot(struct fex_list_entry *entry, VmafFeatureExtractor *fex,
                               VmafDictionary *opts_dict, VmafFeatureExtractorContext **fex_ctx)
{
    for (int i = 0; i < atomic_load(&entry->capacity); i++) {
        int err = ctx_pool_ensure_slot_ctx(entry, i, fex, opts_dict);
        if (err)
            return err;
        if (!entry->ctx_list[i].in_use) {
            *fex_ctx = entry->ctx_list[i].fex_ctx;
            entry->ctx_list[i].in_use = true;
            break;
        }
    }
    atomic_fetch_add(&entry->in_use, 1);
    return 0;
}

int vmaf_fex_ctx_pool_aquire(VmafFeatureExtractorContextPool *pool, VmafFeatureExtractor *fex,
                             VmafDictionary *opts_dict, VmafFeatureExtractorContext **fex_ctx)
{
    if (!pool)
        return -EINVAL;
    if (!fex)
        return -EINVAL;
    if (!fex_ctx)
        return -EINVAL;

    pthread_mutex_lock(&(pool->lock));
    int err = 0;

    struct fex_list_entry *entry = get_fex_list_entry(pool, fex, opts_dict);
    if (!entry) {
        err = -EINVAL;
        goto unlock;
    }

    while (atomic_load(&entry->capacity) == atomic_load(&entry->in_use))
        pthread_cond_wait(&(entry->full), &(pool->lock));

    err = ctx_pool_claim_slot(entry, fex, opts_dict, fex_ctx);

unlock:
    pthread_mutex_unlock(&(pool->lock));
    return err;
}

int vmaf_fex_ctx_pool_release(VmafFeatureExtractorContextPool *pool,
                              VmafFeatureExtractorContext *fex_ctx)
{
    if (!pool)
        return -EINVAL;
    if (!fex_ctx)
        return -EINVAL;

    pthread_mutex_lock(&(pool->lock));
    int err = 0;

    VmafFeatureExtractor *fex = fex_ctx->fex;
    struct fex_list_entry *entry = NULL;
    for (unsigned i = 0; i < pool->cnt; i++) {
        if (!strcmp(fex->name, pool->fex_list[i].fex->name) &&
            !vmaf_dictionary_compare(fex_ctx->opts_dict, pool->fex_list[i].opts_dict)) {
            entry = &pool->fex_list[i];
            break;
        }
    }

    if (!entry) {
        err = -EINVAL;
        goto unlock;
    }

    for (int i = 0; i < atomic_load(&entry->capacity); i++) {
        if (fex_ctx == entry->ctx_list[i].fex_ctx) {
            entry->ctx_list[i].in_use = false;
            atomic_fetch_sub(&entry->in_use, 1);
            pthread_cond_signal(&(entry->full));
            goto unlock;
        }
    }
    err = -EINVAL;

unlock:
    pthread_mutex_unlock(&(pool->lock));
    return err;
}

int vmaf_fex_ctx_pool_flush(VmafFeatureExtractorContextPool *pool,
                            VmafFeatureCollector *feature_collector)
{
    if (!pool)
        return -EINVAL;
    if (!pool->fex_list)
        return -EINVAL;
    pthread_mutex_lock(&(pool->lock));

    for (unsigned i = 0; i < pool->cnt; i++) {
        VmafFeatureExtractor *fex = pool->fex_list[i].fex;
        if (!(fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL))
            continue;
        for (int j = 0; j < atomic_load(&pool->fex_list[i].capacity); j++) {
            VmafFeatureExtractorContext *fex_ctx = pool->fex_list[i].ctx_list[j].fex_ctx;
            if (!fex_ctx)
                continue;
            vmaf_feature_extractor_context_flush(fex_ctx, feature_collector);
        }
    }

    pthread_mutex_unlock(&(pool->lock));
    return 0;
}

int vmaf_fex_ctx_pool_destroy(VmafFeatureExtractorContextPool *pool)
{
    if (!pool)
        return -EINVAL;
    if (!pool->fex_list)
        goto free_pool;
    pthread_mutex_lock(&(pool->lock));

    for (unsigned i = 0; i < pool->cnt; i++) {
        if (!pool->fex_list[i].ctx_list)
            continue;
        for (int j = 0; j < atomic_load(&pool->fex_list[i].capacity); j++) {
            VmafFeatureExtractorContext *fex_ctx = pool->fex_list[i].ctx_list[j].fex_ctx;
            if (!fex_ctx)
                continue;
            vmaf_feature_extractor_context_close(fex_ctx);
            vmaf_feature_extractor_context_destroy(fex_ctx);
            vmaf_dictionary_free(&pool->fex_list[i].opts_dict);
        }
        free(pool->fex_list[i].ctx_list);
    }
    free(pool->fex_list);

free_pool:
    free(pool);
    return 0;
}
