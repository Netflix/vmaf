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

#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
/* MSVC/UCRT provides isatty / fileno via <io.h> under the MSVC-prefixed
 * names _isatty / _fileno; the POSIX-style aliases stay available for
 * source portability. MinGW ships <unistd.h>, so this split is strictly
 * MSVC / clang-cl. */
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

#include "cli_parse.h"
#include "spinner.h"
#include "vidinput.h"

#include "libvmaf/picture.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/dnn.h"
#ifdef HAVE_CUDA
#include "libvmaf/libvmaf_cuda.h"
#endif
#ifdef HAVE_SYCL
#include "libvmaf/libvmaf_sycl.h"
#endif
#ifdef HAVE_VULKAN
#include "libvmaf/libvmaf_vulkan.h"
#endif
#ifdef HAVE_HIP
#include "libvmaf/libvmaf_hip.h"
#endif
#ifdef HAVE_METAL
#include "libvmaf/libvmaf_metal.h"
#endif

static enum VmafPixelFormat pix_fmt_map(int pf)
{
    switch (pf) {
    case PF_420:
        return VMAF_PIX_FMT_YUV420P;
    case PF_422:
        return VMAF_PIX_FMT_YUV422P;
    case PF_444:
        return VMAF_PIX_FMT_YUV444P;
    default:
        return VMAF_PIX_FMT_UNKNOWN;
    }
}

/* Validate per-video constraints that do not require comparing the two streams:
 * supported bitdepth range and positive (non-zero) frame dimensions. */
static int validate_video_info(const video_input_info *info)
{
    int err_cnt = 0;

    if (info->depth < 8 || info->depth > 16) {
        (void)fprintf(stderr, "unsupported bitdepth: %d\n", info->depth);
        err_cnt++;
    }

    /* A zero-width or zero-height frame will produce a divide-by-zero or
     * zero-stride allocation in downstream code. */
    if (info->frame_w <= 0 || info->frame_h <= 0) {
        (void)fprintf(stderr, "non-positive dimensions: %dx%d\n", info->frame_w, info->frame_h);
        err_cnt++;
    }

    return err_cnt;
}

/* Chroma-subsampled formats require even dimensions on the subsampled axes so
 * that the chroma planes contain whole pixels.  PF_420 subsamples both X and
 * Y; PF_422 subsamples X only. */
static int validate_chroma_alignment(const video_input_info *info)
{
    int err_cnt = 0;

    if (info->pixel_fmt == PF_420 || info->pixel_fmt == PF_422) {
        if (info->frame_w % 2 != 0) {
            (void)fprintf(stderr, "odd width %d not allowed for chroma-subsampled format\n",
                          info->frame_w);
            err_cnt++;
        }
    }
    if (info->pixel_fmt == PF_420) {
        if (info->frame_h % 2 != 0) {
            (void)fprintf(stderr, "odd height %d not allowed for 4:2:0 format\n", info->frame_h);
            err_cnt++;
        }
    }

    return err_cnt;
}

static int validate_videos(video_input *vid1, video_input *vid2, bool common_bitdepth)
{
    int err_cnt = 0;

    video_input_info info1;
    video_input_info info2;
    video_input_get_info(vid1, &info1);
    video_input_get_info(vid2, &info2);

    if ((info1.frame_w != info2.frame_w) || (info1.frame_h != info2.frame_h)) {
        (void)fprintf(stderr, "dimensions do not match: %dx%d, %dx%d\n", info1.frame_w,
                      info1.frame_h, info2.frame_w, info2.frame_h);
        err_cnt++;
    }

    if (info1.pixel_fmt != info2.pixel_fmt) {
        (void)fprintf(stderr, "pixel formats do not match: %d, %d\n", info1.pixel_fmt,
                      info2.pixel_fmt);
        err_cnt++;
    }

    if (!pix_fmt_map(info1.pixel_fmt) || !pix_fmt_map(info2.pixel_fmt)) {
        (void)fprintf(stderr, "unsupported pixel format: %d\n", info1.pixel_fmt);
        err_cnt++;
    }

    if (!common_bitdepth && info1.depth != info2.depth) {
        (void)fprintf(stderr, "bitdepths do not match: %d, %d\n", info1.depth, info2.depth);
        err_cnt++;
    }

    err_cnt += validate_video_info(&info1);
    err_cnt += validate_video_info(&info2);
    err_cnt += validate_chroma_alignment(&info1);

    return err_cnt;
}

/* Copy video input data to picture buffer. The four bit-depth × component
 * branches (8-bit Y/U/V, 10-bit Y/U/V, 16-bit packed) duplicate the per-row
 * loop with different per-sample casts; folding them through a function
 * pointer would cost a per-row indirect call on every frame, so the
 * branches stay inline. The nesting-level warning is structural to YUV
 * (plane × row × column) — splitting wouldn't reduce it
 * (ADR-0141 §2 load-bearing invariant: per-frame indirect-call cost;
 * T7-5 sweep closeout — ADR-0278).
 */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
static void copy_picture_data(VmafPicture *pic, video_input_ycbcr ycbcr, video_input_info *info,
                              int depth)
{
    if (info->depth == depth) {
        if (info->depth == 8) {
            for (unsigned i = 0; i < 3; i++) {
                int xdec = i && !(info->pixel_fmt & 1);
                int ydec = i && !(info->pixel_fmt & 2);
                uint8_t *ycbcr_data = ycbcr[i].data +
                                      (size_t)(info->pic_y >> ydec) * ycbcr[i].stride +
                                      (info->pic_x >> xdec);
                uint8_t *pic_data = pic->data[i];

                for (unsigned j = 0; j < pic->h[i]; j++) {
                    memcpy(pic_data, ycbcr_data, sizeof(*pic_data) * pic->w[i]);
                    pic_data += pic->stride[i];
                    ycbcr_data += ycbcr[i].stride;
                }
            }
        } else {
            for (unsigned i = 0; i < 3; i++) {
                int xdec = i && !(info->pixel_fmt & 1);
                int ydec = i && !(info->pixel_fmt & 2);
                uint16_t *ycbcr_data = (uint16_t *)ycbcr[i].data +
                                       (size_t)(info->pic_y >> ydec) * (ycbcr[i].stride / 2) +
                                       (info->pic_x >> xdec);
                uint16_t *pic_data = pic->data[i];

                for (unsigned j = 0; j < pic->h[i]; j++) {
                    memcpy(pic_data, ycbcr_data, sizeof(*pic_data) * pic->w[i]);
                    pic_data += pic->stride[i] / 2;
                    ycbcr_data += ycbcr[i].stride / 2;
                }
            }
        }
    } else if (depth > 8) {
        // unequal bit-depth
        // therefore depth must be > 8 since we do not support depth < 8
        int left_shift = depth - info->depth;
        if (info->depth == 8) {
            for (unsigned i = 0; i < 3; i++) {
                int xdec = i && !(info->pixel_fmt & 1);
                int ydec = i && !(info->pixel_fmt & 2);
                uint8_t *ycbcr_data = ycbcr[i].data +
                                      (size_t)(info->pic_y >> ydec) * ycbcr[i].stride +
                                      (info->pic_x >> xdec);
                uint16_t *pic_data = (uint16_t *)pic->data[i];

                for (unsigned j = 0; j < pic->h[i]; j++) {
                    for (unsigned k = 0; k < pic->w[i]; k++) {
                        pic_data[k] = ycbcr_data[k] << left_shift;
                    }
                    pic_data += pic->stride[i] / 2;
                    ycbcr_data += ycbcr[i].stride;
                }
            }
        } else {
            for (unsigned i = 0; i < 3; i++) {
                int xdec = i && !(info->pixel_fmt & 1);
                int ydec = i && !(info->pixel_fmt & 2);
                uint16_t *ycbcr_data = (uint16_t *)ycbcr[i].data +
                                       (size_t)(info->pic_y >> ydec) * (ycbcr[i].stride / 2) +
                                       (info->pic_x >> xdec);
                uint16_t *pic_data = pic->data[i];

                for (unsigned j = 0; j < pic->h[i]; j++) {
                    for (unsigned k = 0; k < pic->w[i]; k++) {
                        pic_data[k] = ycbcr_data[k] << left_shift;
                    }
                    pic_data += pic->stride[i] / 2;
                    ycbcr_data += ycbcr[i].stride / 2;
                }
            }
        }
    }
}

static int fetch_picture(VmafContext *vmaf, video_input *vid, VmafPicture *pic, int depth)
{
    int ret;
    video_input_ycbcr ycbcr;
    video_input_info info;

    ret = video_input_fetch_frame(vid, ycbcr, NULL);
    if (ret < 1)
        return !ret;

    video_input_get_info(vid, &info);

    ret = vmaf_fetch_preallocated_picture(vmaf, pic);
    if (ret) {
        (void)fprintf(stderr, "problem fetching picture from pool.\n");
        return -1;
    }

    copy_picture_data(pic, ycbcr, &info, depth);
    return 0;
}

/* Allocate and zero the three parallel arrays the CLI uses to track loaded
 * models, model-collection slots, and the corresponding labels for status
 * output. The output pointers are zero-initialised, so the cleanup block
 * can call vmaf_model_destroy() / vmaf_model_collection_destroy() safely
 * over arrays that may be partially populated.
 */
static int allocate_model_arrays(unsigned model_cnt, VmafModel ***model,
                                 VmafModelCollection ***model_collection,
                                 const char ***model_collection_label)
{
    const size_t model_sz = sizeof(**model) * model_cnt;
    *model = (VmafModel **)malloc(model_sz);
    if (!*model)
        return -1;
    memset((void *)*model, 0, model_sz);

    const size_t model_collection_sz = sizeof(**model_collection) * model_cnt;
    *model_collection = (VmafModelCollection **)malloc(model_collection_sz);
    if (!*model_collection)
        return -1;
    memset((void *)*model_collection, 0, model_collection_sz);

    const size_t label_sz = sizeof(**model_collection_label) * model_cnt;
    *model_collection_label = (const char **)malloc(label_sz);
    if (!*model_collection_label)
        return -1;
    memset((void *)*model_collection_label, 0, label_sz);

    return 0;
}

/* Helper: pick the human-readable label (version preferred over path) for
 * the given model-config entry, used in error messages.
 */
static const char *model_label(const CLISettings *c, unsigned i)
{
    return c->model_config[i].version ? c->model_config[i].version : c->model_config[i].path;
}

/* Initialise a model-collection slot for entry `i`. The caller passes the
 * current `*slot` index; on any failure path this helper bumps `*slot`
 * before returning so the caller's cleanup loop unwinds the partially
 * initialised entry. Returns 0 on success.
 */
static int load_model_collection_entry(VmafContext *vmaf, CLISettings *c, unsigned i,
                                       VmafModel **model, VmafModelCollection **model_collection,
                                       const char **model_collection_label, unsigned *slot)
{
    int err;

    if (c->model_config[i].version) {
        err = vmaf_model_collection_load(&model[i], &model_collection[*slot],
                                         &c->model_config[i].cfg, c->model_config[i].version);
    } else {
        err = vmaf_model_collection_load_from_path(
            &model[i], &model_collection[*slot], &c->model_config[i].cfg, c->model_config[i].path);
    }

    if (err) {
        (void)fprintf(stderr, "problem loading model: %s\n", model_label(c, i));
        return -1;
    }

    model_collection_label[*slot] = model_label(c, i);

    for (unsigned j = 0; j < c->model_config[i].overload_cnt; j++) {
        err = vmaf_model_collection_feature_overload(
            model[i], &model_collection[*slot], c->model_config[i].feature_overload[j].name,
            c->model_config[i].feature_overload[j].opts_dict);
        if (err) {
            (void)fprintf(stderr,
                          "problem overloading feature extractors from model collection: %s\n",
                          model_label(c, i));
            (*slot)++;
            return -1;
        }
    }

    err = vmaf_use_features_from_model_collection(vmaf, model_collection[*slot]);
    if (err) {
        (void)fprintf(stderr, "problem loading feature extractors from model collection: %s\n",
                      model_label(c, i));
        (*slot)++;
        return -1;
    }

    (*slot)++;
    return 0;
}

/* Load a single model entry from the CLI configuration. Handles the model
 * vs model-collection fallback that the `--model` option's overloaded
 * semantics require.
 */
static int load_one_model_entry(VmafContext *vmaf, CLISettings *c, unsigned i, VmafModel **model,
                                VmafModelCollection **model_collection,
                                const char **model_collection_label, unsigned *model_collection_cnt)
{
    int err;

    if (c->model_config[i].version) {
        err = vmaf_model_load(&model[i], &c->model_config[i].cfg, c->model_config[i].version);
    } else {
        err =
            vmaf_model_load_from_path(&model[i], &c->model_config[i].cfg, c->model_config[i].path);
    }

    /* `--model` is overloaded: if a single-model load fails, fall back to
     * loading the same identifier as a model collection.
     */
    if (err) {
        return load_model_collection_entry(vmaf, c, i, model, model_collection,
                                           model_collection_label, model_collection_cnt);
    }

    for (unsigned j = 0; j < c->model_config[i].overload_cnt; j++) {
        err = vmaf_model_feature_overload(model[i], c->model_config[i].feature_overload[j].name,
                                          c->model_config[i].feature_overload[j].opts_dict);
        if (err) {
            (void)fprintf(stderr, "problem overloading feature extractors from model: %s\n",
                          model_label(c, i));
            return -1;
        }
    }

    err = vmaf_use_features_from_model(vmaf, model[i]);
    if (err) {
        (void)fprintf(stderr, "problem loading feature extractors from model: %s\n",
                      model_label(c, i));
        return -1;
    }

    return 0;
}

/* Open both reference and distorted input streams (raw YUV via raw_input_open
 * when --use_yuv is set, otherwise the codec auto-detection path via
 * video_input_open). On success transfers FILE* ownership from *file_ref/dist
 * to the corresponding video_input and zeros the pointers so the cleanup
 * fclose() doesn't double-close. Sets *vid_ref_open / *vid_dist_open to true
 * for cleanup unwinding. Returns 0 on success, -1 on any failure (caller
 * should treat as fatal and `goto cleanup`).
 */
static int open_input_videos(const CLISettings *c, FILE **file_ref, FILE **file_dist,
                             video_input *vid_ref, video_input *vid_dist, bool *vid_ref_open,
                             bool *vid_dist_open)
{
    int err;

    if (c->use_yuv) {
        err = raw_input_open(vid_ref, *file_ref, c->width, c->height, c->pix_fmt, c->bitdepth);
    } else {
        err = video_input_open(vid_ref, *file_ref);
    }
    if (err) {
        (void)fprintf(stderr, "problem with reference file: %s\n", c->path_ref);
        return -1;
    }
    *vid_ref_open = true;
    *file_ref = NULL; /* ownership transferred to vid_ref */

    if (c->use_yuv) {
        err = raw_input_open(vid_dist, *file_dist, c->width, c->height, c->pix_fmt, c->bitdepth);
    } else {
        err = video_input_open(vid_dist, *file_dist);
    }
    if (err) {
        (void)fprintf(stderr, "problem with distorted file: %s\n", c->path_dist);
        return -1;
    }
    *vid_dist_open = true;
    *file_dist = NULL; /* ownership transferred to vid_dist */

    err = validate_videos(vid_ref, vid_dist, c->common_bitdepth);
    if (err) {
        (void)fprintf(stderr, "videos are incompatible, %d %s.\n", err,
                      err == 1 ? "problem" : "problems");
        return -1;
    }

    return 0;
}

/* Initialise the GPU backends in declared priority order: SYCL first
 * (preferred when --sycl_device or --gpumask is set), CUDA second
 * (consulted only if SYCL was not activated), Vulkan last (explicit
 * --vulkan_device opt-in, host-pic only). On a hard backend-import failure
 * returns -1 so the caller can `goto cleanup`; soft init failures
 * (state_init returning non-zero) silently fall back to CPU. The
 * sycl/vulkan active flags + state pointers are passed by reference so
 * the cleanup block can free them.
 *
 * The function is intentionally kept in a single TU even though three
 * #ifdef-guarded backend stanzas push the line count just past the
 * 60-line threshold. Splitting into per-backend helpers would multiply
 * the `#if defined(HAVE_X)` decoration without making the activation
 * priority chain (SYCL > CUDA > Vulkan) any clearer to a reader
 * (ADR-0141 §2 load-bearing invariant: backend-priority chain
 * readability + #ifdef discipline; T7-5 sweep closeout — ADR-0278).
 */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
static int init_gpu_backends(VmafContext *vmaf, const CLISettings *c
#ifdef HAVE_SYCL
                             ,
                             VmafSyclState **sycl_state, bool *sycl_active
#endif
#ifdef HAVE_VULKAN
                             ,
                             VmafVulkanState **vulkan_state, bool *vulkan_active
#endif
#ifdef HAVE_HIP
                             ,
                             VmafHipState **hip_state, bool *hip_active
#endif
#ifdef HAVE_METAL
                             ,
                             VmafMetalState **metal_state, bool *metal_active
#endif
)
{
    int err;
    (void)vmaf;
    (void)c;
    (void)err;

    // GPU backend initialization: each backend activates only when its
    // specific flag is passed.  --gpumask enables the preferred backend
    // (SYCL > CUDA).  --sycl_device selects
    // that specific backend.  No flag = CPU only.
#ifdef HAVE_SYCL
    VmafSyclConfiguration sycl_cfg = {
        .device_index = c->sycl_device >= 0 ? c->sycl_device : 0,
    };
    if ((c->sycl_device >= 0 || c->use_gpumask) && !c->no_sycl) {
        err = vmaf_sycl_state_init(sycl_state, sycl_cfg);
        if (err) {
            (void)fprintf(stderr, "problem during vmaf_sycl_state_init, using CPU\n");
        } else {
            err = vmaf_sycl_import_state(vmaf, *sycl_state);
            if (err) {
                (void)fprintf(stderr, "problem during vmaf_sycl_import_state\n");
                return -1;
            }
            *sycl_active = true;
        }
    }
#endif
#ifdef HAVE_CUDA
    bool cuda_active = false;
    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = {0};
    if (c->use_gpumask && !c->no_cuda
#ifdef HAVE_SYCL
        && !*sycl_active
#endif
    ) {
        err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
        if (err) {
            (void)fprintf(stderr, "problem during vmaf_cuda_state_init, using CPU\n");
        } else {
            err |= vmaf_cuda_import_state(vmaf, cu_state);
            if (err) {
                (void)fprintf(stderr, "problem during vmaf_cuda_import_state\n");
                return -1;
            }
            cuda_active = true;
        }
    }
    (void)cuda_active;
#endif

#ifdef HAVE_VULKAN
    /* Vulkan opt-in: explicit --vulkan_device only. Unlike SYCL/CUDA
     * the gpumask gate is not consulted; Vulkan is host-pic only and
     * has no restricted-mode semantics yet. */
    VmafVulkanConfiguration vulkan_cfg = {
        .device_index = c->vulkan_device,
        .enable_validation = 0,
    };
    if (c->vulkan_device >= 0 && !c->no_vulkan) {
        err = vmaf_vulkan_state_init(vulkan_state, vulkan_cfg);
        if (err) {
            (void)fprintf(stderr, "problem during vmaf_vulkan_state_init (%d), using CPU\n", err);
        } else {
            err = vmaf_vulkan_import_state(vmaf, *vulkan_state);
            if (err) {
                (void)fprintf(stderr, "problem during vmaf_vulkan_import_state\n");
                return -1;
            }
            *vulkan_active = true;
        }
    }
    (void)*vulkan_active;
#endif

#ifdef HAVE_HIP
    /* HIP opt-in: explicit --hip_device only. Same lifetime model as
     * SYCL + Vulkan — state is passed back by reference so the cleanup
     * block can free it after vmaf_close(). */
    VmafHipConfiguration hip_cfg = {
        .device_index = c->hip_device,
        .flags = 0,
    };
    if (c->hip_device >= 0 && !c->no_hip) {
        err = vmaf_hip_state_init(hip_state, hip_cfg);
        if (err) {
            (void)fprintf(stderr, "problem during vmaf_hip_state_init (%d), using CPU\n", err);
        } else {
            err = vmaf_hip_import_state(vmaf, *hip_state);
            if (err) {
                (void)fprintf(stderr, "problem during vmaf_hip_import_state\n");
                return -1;
            }
            *hip_active = true;
        }
    }
    (void)*hip_active;
#endif

#ifdef HAVE_METAL
    /* Metal opt-in: explicit --metal_device only. macOS-only; on non-
     * Apple hosts vmaf_metal_state_init returns -ENODEV and the CLI
     * falls back to CPU. Same state-lifetime model as SYCL/Vulkan. */
    VmafMetalConfiguration metal_cfg = {
        .device_index = c->metal_device,
        .flags = 0,
    };
    if (c->metal_device >= 0 && !c->no_metal) {
        err = vmaf_metal_state_init(metal_state, metal_cfg);
        if (err) {
            (void)fprintf(stderr, "problem during vmaf_metal_state_init (%d), using CPU\n", err);
        } else {
            err = vmaf_metal_import_state(vmaf, *metal_state);
            if (err) {
                (void)fprintf(stderr, "problem during vmaf_metal_import_state\n");
                return -1;
            }
            *metal_active = true;
        }
    }
    (void)*metal_active;
#endif

    return 0;
}

/* Translate the textual --tiny-device flag (cpu / cuda / openvino /
 * coreml / coreml-ane / coreml-gpu / coreml-cpu / openvino-npu /
 * openvino-cpu / openvino-gpu / rocm) into the corresponding
 * VmafDnnDevice enum. The coreml-* keywords pin the CoreML EP to a
 * single MLComputeUnits value (see ADR-0365); plain `coreml` lets
 * CoreML auto-route across compute units. The openvino-* keywords pin
 * the OpenVINO EP to a single device type with no fallback (see
 * Research-0031); plain `openvino` keeps the GPU→CPU fallback chain.
 * Unknown values fall back to VMAF_DNN_DEVICE_AUTO so the runtime
 * picks a default.
 */
static VmafDnnDevice resolve_tiny_device(const char *name)
{
    if (!name)
        return VMAF_DNN_DEVICE_AUTO;
    if (!strcmp(name, "cpu"))
        return VMAF_DNN_DEVICE_CPU;
    if (!strcmp(name, "cuda"))
        return VMAF_DNN_DEVICE_CUDA;
    if (!strcmp(name, "openvino"))
        return VMAF_DNN_DEVICE_OPENVINO;
    if (!strcmp(name, "coreml"))
        return VMAF_DNN_DEVICE_COREML;
    if (!strcmp(name, "coreml-ane"))
        return VMAF_DNN_DEVICE_COREML_ANE;
    if (!strcmp(name, "coreml-gpu"))
        return VMAF_DNN_DEVICE_COREML_GPU;
    if (!strcmp(name, "coreml-cpu"))
        return VMAF_DNN_DEVICE_COREML_CPU;
    if (!strcmp(name, "openvino-npu"))
        return VMAF_DNN_DEVICE_OPENVINO_NPU;
    if (!strcmp(name, "openvino-cpu"))
        return VMAF_DNN_DEVICE_OPENVINO_CPU;
    if (!strcmp(name, "openvino-gpu"))
        return VMAF_DNN_DEVICE_OPENVINO_GPU;
    if (!strcmp(name, "rocm"))
        return VMAF_DNN_DEVICE_ROCM;
    return VMAF_DNN_DEVICE_AUTO;
}

/* Configure the tiny-AI (DNN) model on the VMAF context when --tiny-model
 * is passed. Performs the optional Sigstore-bundle verification (T6-9 /
 * ADR-0211) before opening the model so a signature failure short-circuits
 * load and never touches ORT. Returns 0 on success, -1 on any failure
 * (caller should treat as fatal and `goto cleanup`).
 */
static int configure_tiny_model(VmafContext *vmaf, const CLISettings *c)
{
    if (!c->tiny_model_path)
        return 0;

    if (!vmaf_dnn_available()) {
        (void)fprintf(stderr,
                      "--tiny-model requested (%s) but libvmaf was built "
                      "without DNN support (-Denable_dnn=disabled).\n",
                      c->tiny_model_path);
        return -1;
    }
    /* T6-9 / ADR-0211 — Sigstore-bundle verification. Runs *before*
     * the model is opened so a verification failure short-circuits
     * load and never touches ORT. Fails closed: missing registry,
     * missing bundle, missing cosign, or any non-zero cosign exit
     * all refuse to proceed. */
    if (c->tiny_model_verify) {
        const int verr = vmaf_dnn_verify_signature(c->tiny_model_path, NULL);
        if (verr != 0) {
            (void)fprintf(stderr,
                          "--tiny-model-verify: signature verification "
                          "failed for %s (errno %d)\n",
                          c->tiny_model_path, -verr);
            return -1;
        }
    }
    VmafDnnConfig dnn_cfg = {
        .device = resolve_tiny_device(c->tiny_device),
        .device_index = 0,
        .threads = c->tiny_threads,
        .fp16_io = c->tiny_fp16,
    };
    int err = vmaf_use_tiny_model(vmaf, c->tiny_model_path, &dnn_cfg);
    if (err) {
        (void)fprintf(stderr, "problem loading tiny model %s: %d\n", c->tiny_model_path, err);
        return -1;
    }

    return 0;
}

/* Skip the first `c->frame_skip_ref` ref frames and `c->frame_skip_dist` dist
 * frames, releasing each one back to the picture pool. fetch_picture() reserves
 * a slot from the preallocated pool, and skipped frames are never handed to
 * vmaf_read_pictures() to release them; without unref the pool is exhausted
 * after N skips and the next fetch blocks indefinitely.
 */
static void skip_initial_frames(VmafContext *vmaf, video_input *vid_ref, video_input *vid_dist,
                                const CLISettings *c, int common_bitdepth)
{
    VmafPicture pic_ref_skip;
    VmafPicture pic_dist_skip;

    for (unsigned i = 0; i < c->frame_skip_ref; i++) {
        if (fetch_picture(vmaf, vid_ref, &pic_ref_skip, common_bitdepth))
            break;
        if (vmaf_picture_unref(&pic_ref_skip))
            (void)fprintf(stderr, "\nproblem during vmaf_picture_unref (skip ref)\n");
    }

    for (unsigned i = 0; i < c->frame_skip_dist; i++) {
        if (fetch_picture(vmaf, vid_dist, &pic_dist_skip, common_bitdepth))
            break;
        if (vmaf_picture_unref(&pic_dist_skip))
            (void)fprintf(stderr, "\nproblem during vmaf_picture_unref (skip dist)\n");
    }
}

/* Drive the main per-frame fetch + process loop. Returns the number of frames
 * successfully consumed (the post-increment `picture_index` value the original
 * inline loop used to compute `picture_index - 1` in pooling). Stops at EOF
 * on either side, on read errors, or when c->frame_cnt is reached.
 */
static unsigned run_frame_loop(VmafContext *vmaf, video_input *vid_ref, video_input *vid_dist,
                               const CLISettings *c, int common_bitdepth, int istty)
{
    float fps = 0.;
    const time_t t0 = clock();
    unsigned picture_index;
    for (picture_index = 0;; picture_index++) {

        if (c->frame_cnt && picture_index >= c->frame_cnt)
            break;

        VmafPicture pic_ref;
        VmafPicture pic_dist;
        int ret1 = fetch_picture(vmaf, vid_ref, &pic_ref, common_bitdepth);
        int ret2 = fetch_picture(vmaf, vid_dist, &pic_dist, common_bitdepth);

        if (ret1 && ret2) {
            break;
        } else if (ret1 < 0 || ret2 < 0) {
            (void)fprintf(stderr, "\nproblem while reading pictures\n");
            break;
        } else if (ret1) {
            (void)fprintf(stderr, "\n\"%s\" ended before \"%s\".\n", c->path_ref, c->path_dist);
            int err_unref = vmaf_picture_unref(&pic_dist);
            if (err_unref)
                (void)fprintf(stderr, "\nproblem during vmaf_picture_unref\n");
            break;
        } else if (ret2) {
            (void)fprintf(stderr, "\n\"%s\" ended before \"%s\".\n", c->path_dist, c->path_ref);
            int err_unref = vmaf_picture_unref(&pic_ref);
            if (err_unref)
                (void)fprintf(stderr, "\nproblem during vmaf_picture_unref\n");
            break;
        }

        if (istty && !c->quiet) {
            if (picture_index > 0 && !(picture_index % 10)) {
                fps = (picture_index + 1) / (((float)clock() - t0) / CLOCKS_PER_SEC);
            }

            (void)fprintf(stderr, "\r%d frame%s %s %.2f FPS\033[K", picture_index + 1,
                          picture_index ? "s" : " ", spinner[picture_index % spinner_length], fps);
            (void)fflush(stderr);
        }

        int err = vmaf_read_pictures(vmaf, &pic_ref, &pic_dist, picture_index);
        if (err) {
            (void)fprintf(stderr, "\nproblem reading pictures\n");
            break;
        }
    }
    if (istty && !c->quiet)
        (void)fprintf(stderr, "\n");

    return picture_index;
}

/* Compute and report pooled VMAF scores for all loaded models and model
 * collections. Called only when c->no_prediction is false. Returns 0 on
 * success, non-zero on the first per-model scoring failure (caller should
 * treat as fatal and `goto cleanup`).
 */
static int report_pooled_scores(VmafContext *vmaf, const CLISettings *c, VmafModel **model,
                                VmafModelCollection **model_collection,
                                const char **model_collection_label, unsigned model_collection_cnt,
                                unsigned picture_index, int istty)
{
    int err = 0;

    for (unsigned i = 0; i < c->model_cnt; i++) {
        double vmaf_score;
        err = vmaf_score_pooled(vmaf, model[i], VMAF_POOL_METHOD_MEAN, &vmaf_score, 0,
                                picture_index - 1);
        if (err) {
            (void)fprintf(stderr, "problem generating pooled VMAF score\n");
            return -1;
        }

        if (istty && (!c->quiet || !c->output_path)) {
            (void)fprintf(stderr, "%s: ",
                          c->model_config[i].version ? c->model_config[i].version :
                                                       c->model_config[i].path);
            (void)fprintf(stderr, c->precision_fmt, vmaf_score);
            (void)fprintf(stderr, "\n");
        }
    }

    for (unsigned i = 0; i < model_collection_cnt; i++) {
        VmafModelCollectionScore score = {0};
        err = vmaf_score_pooled_model_collection(vmaf, model_collection[i], VMAF_POOL_METHOD_MEAN,
                                                 &score, 0, picture_index - 1);
        if (err) {
            (void)fprintf(stderr, "problem generating pooled VMAF score\n");
            return -1;
        }

        switch (score.type) {
        case VMAF_MODEL_COLLECTION_SCORE_BOOTSTRAP:
            if (istty && (!c->quiet || !c->output_path)) {
                (void)fprintf(stderr, "%s: ", model_collection_label[i]);
                (void)fprintf(stderr, c->precision_fmt, score.bootstrap.bagging_score);
                (void)fprintf(stderr, ", ci.p95: [");
                (void)fprintf(stderr, c->precision_fmt, score.bootstrap.ci.p95.lo);
                (void)fprintf(stderr, ", ");
                (void)fprintf(stderr, c->precision_fmt, score.bootstrap.ci.p95.hi);
                (void)fprintf(stderr, "], stddev: ");
                (void)fprintf(stderr, c->precision_fmt, score.bootstrap.stddev);
                (void)fprintf(stderr, "\n");
            }
            break;
        default:
            break;
        }
    }

    return 0;
}

/* CLI driver: orchestrates input opening, VMAF context init, GPU backend
 * activation, model loading, frame loop, score reporting, and cleanup.
 * The function is structured around a single goto-cleanup block (not
 * RAII-style helpers) because each subsystem owns a distinct cleanup
 * primitive (fclose / video_input_close / vmaf_close /
 * vmaf_*_state_free / vmaf_model_destroy / cli_free) that must run in
 * reverse-init order on every exit path. T7-5 sweep extracted the eight
 * largest sub-blocks into named helpers (open_input_videos,
 * init_gpu_backends, allocate_model_arrays, load_one_model_entry,
 * configure_tiny_model, skip_initial_frames, run_frame_loop,
 * report_pooled_scores). The remaining body is the cleanup-ownership
 * spine plus inter-step glue; further extraction would require
 * pointer-aliasing the cleanup-relevant locals through helper signatures
 * and obscure the unwind chain
 * (ADR-0141 §2 load-bearing invariant: goto-cleanup ownership chain;
 * T7-5 sweep closeout — ADR-0278; ADR-0146 prior sweep precedent).
 */
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
int main(int argc, char *argv[])
{
    int err = 0;
    int ret = 0;
    const int istty = isatty(fileno(stderr));

    CLISettings c;
    cli_parse(argc, argv, &c);

    FILE *file_ref = NULL;
    FILE *file_dist = NULL;
    bool vid_ref_open = false;
    bool vid_dist_open = false;
    video_input vid_ref = {0};
    video_input vid_dist = {0};
    VmafContext *vmaf = NULL;
    VmafModel **model = NULL;
    VmafModelCollection **model_collection = NULL;
    const char **model_collection_label = NULL;
    unsigned model_collection_cnt = 0;
#ifdef HAVE_SYCL
    bool sycl_active = false;
    VmafSyclState *sycl_state = NULL;
#endif
#ifdef HAVE_VULKAN
    bool vulkan_active = false;
    VmafVulkanState *vulkan_state = NULL;
#endif
#ifdef HAVE_HIP
    bool hip_active = false;
    VmafHipState *hip_state = NULL;
#endif
#ifdef HAVE_METAL
    bool metal_active = false;
    VmafMetalState *metal_state = NULL;
#endif

    if (istty && !c.quiet) {
        (void)fprintf(stderr, "VMAF version %s\n", vmaf_version());
    }

    file_ref = fopen(c.path_ref, "rb");
    if (!file_ref) {
        (void)fprintf(stderr, "could not open file: %s\n", c.path_ref);
        ret = -1;
        goto cleanup;
    }

    file_dist = fopen(c.path_dist, "rb");
    if (!file_dist) {
        (void)fprintf(stderr, "could not open file: %s\n", c.path_dist);
        ret = -1;
        goto cleanup;
    }

    if (open_input_videos(&c, &file_ref, &file_dist, &vid_ref, &vid_dist, &vid_ref_open,
                          &vid_dist_open)) {
        ret = -1;
        goto cleanup;
    }

    int common_bitdepth;
    if (c.use_yuv) {
        common_bitdepth = c.bitdepth;
    } else {
        video_input_info info1;
        video_input_info info2;
        video_input_get_info(&vid_ref, &info1);
        video_input_get_info(&vid_dist, &info2);
        common_bitdepth = info1.depth > info2.depth ? info1.depth : info2.depth;
    }

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = c.thread_cnt,
        .n_subsample = c.subsample,
        .cpumask = c.cpumask,
        .gpumask = c.gpumask,
    };

    err = vmaf_init(&vmaf, cfg);
    if (err) {
        (void)fprintf(stderr, "problem initializing VMAF context\n");
        ret = -1;
        goto cleanup;
    }

    if (init_gpu_backends(vmaf, &c
#ifdef HAVE_SYCL
                          ,
                          &sycl_state, &sycl_active
#endif
#ifdef HAVE_VULKAN
                          ,
                          &vulkan_state, &vulkan_active
#endif
#ifdef HAVE_HIP
                          ,
                          &hip_state, &hip_active
#endif
#ifdef HAVE_METAL
                          ,
                          &metal_state, &metal_active
#endif
                          )) {
        ret = -1;
        goto cleanup;
    }

    // Preallocate picture pool to avoid allocation overhead
    video_input_info info;
    video_input_get_info(&vid_ref, &info);

    VmafPictureConfiguration pic_cfg = {
        .pic_params =
            {
                .w = info.pic_w,
                .h = info.pic_h,
                .bpc = common_bitdepth,
                .pix_fmt = pix_fmt_map(info.pixel_fmt),
            },
        /* Liveness budget per frame:
         *   2  — ref + dist currently held by the CLI fetch/process step
         *   1  — `vmaf->prev_ref` keeps the previous frame's ref picture
         *        live across the frame boundary (for motion features)
         *   2*thread_cnt — worker threads may hold (ref, dist) on in-flight
         *        frames that haven't finished processing yet
         * The `+ 1` term covers prev_ref uniformly. Undersizing deadlocks
         * vmaf_picture_pool_fetch on frame N+1. */
        .pic_cnt = 2 * (c.thread_cnt + 1) + 1,
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    if (err) {
        (void)fprintf(stderr, "problem during vmaf_preallocate_pictures\n");
        ret = -1;
        goto cleanup;
    }

    if (istty && !c.quiet) {
        (void)fprintf(stderr, "picture pool: %d pictures pre-allocated\n", pic_cfg.pic_cnt);
    }

    if (allocate_model_arrays(c.model_cnt, &model, &model_collection, &model_collection_label)) {
        ret = -1;
        goto cleanup;
    }

    for (unsigned i = 0; i < c.model_cnt; i++) {
        if (load_one_model_entry(vmaf, &c, i, model, model_collection, model_collection_label,
                                 &model_collection_cnt)) {
            ret = -1;
            goto cleanup;
        }
    }

    for (unsigned i = 0; i < c.feature_cnt; i++) {
        err = vmaf_use_feature(vmaf, c.feature_cfg[i].name, c.feature_cfg[i].opts_dict);
        if (err) {
            (void)fprintf(stderr, "problem loading feature extractor: %s\n", c.feature_cfg[i].name);
            ret = -1;
            goto cleanup;
        }
    }

    if (configure_tiny_model(vmaf, &c)) {
        ret = -1;
        goto cleanup;
    }

    skip_initial_frames(vmaf, &vid_ref, &vid_dist, &c, common_bitdepth);

    unsigned picture_index = run_frame_loop(vmaf, &vid_ref, &vid_dist, &c, common_bitdepth, istty);

    err |= vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) {
        (void)fprintf(stderr, "problem flushing context\n");
        ret = err;
        goto cleanup;
    }

    if (!c.no_prediction) {
        ret = report_pooled_scores(vmaf, &c, model, model_collection, model_collection_label,
                                   model_collection_cnt, picture_index, istty);
        if (ret)
            goto cleanup;
    }

    if (c.output_path)
        vmaf_write_output_with_format(vmaf, c.output_path, c.output_fmt, c.precision_fmt);

    ret = err;

cleanup:
    if (model) {
        for (unsigned i = 0; i < c.model_cnt; i++)
            vmaf_model_destroy(model[i]);
        free((void *)model);
    }
    if (model_collection) {
        for (unsigned i = 0; i < model_collection_cnt; i++)
            vmaf_model_collection_destroy(model_collection[i]);
        free((void *)model_collection);
    }
    free((void *)model_collection_label);
    if (vmaf)
        vmaf_close(vmaf);
#ifdef HAVE_SYCL
    if (sycl_active)
        vmaf_sycl_state_free(&sycl_state);
#endif
#ifdef HAVE_VULKAN
    if (vulkan_state)
        vmaf_vulkan_state_free(&vulkan_state);
#endif
#ifdef HAVE_HIP
    if (hip_state)
        vmaf_hip_state_free(&hip_state);
#endif
#ifdef HAVE_METAL
    if (metal_state)
        vmaf_metal_state_free(&metal_state);
#endif
    if (vid_dist_open)
        video_input_close(&vid_dist);
    if (vid_ref_open)
        video_input_close(&vid_ref);
    if (file_dist)
        (void)fclose(file_dist);
    if (file_ref)
        (void)fclose(file_ref);
    cli_free(&c);
    return ret;
}
