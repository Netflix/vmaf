#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

static int validate_videos(video_input *vid1, video_input *vid2, bool common_bitdepth)
{
    int err_cnt = 0;

    video_input_info info1, info2;
    video_input_get_info(vid1, &info1);
    video_input_get_info(vid2, &info2);

    if ((info1.frame_w != info2.frame_w) || (info1.frame_h != info2.frame_h)) {
        fprintf(stderr, "dimensions do not match: %dx%d, %dx%d\n",
                info1.frame_w, info1.frame_h, info2.frame_w, info2.frame_h);
        err_cnt++;
    }

    if (info1.pixel_fmt != info2.pixel_fmt) {
        fprintf(stderr, "pixel formats do not match: %d, %d\n",
                info1.pixel_fmt, info2.pixel_fmt);
        err_cnt++;
    }

    if (!pix_fmt_map(info1.pixel_fmt) || !pix_fmt_map(info2.pixel_fmt)) {
        fprintf(stderr, "unsupported pixel format: %d\n", info1.pixel_fmt);
        err_cnt++;
    }

    if (!common_bitdepth && info1.depth != info2.depth) {
        fprintf(stderr, "bitdepths do not match: %d, %d\n",
                info1.depth, info2.depth);
        err_cnt++;
    }

    if (info1.depth < 8 || info1.depth > 16) {
        fprintf(stderr, "unsupported bitdepth: %d\n", info1.depth);
        err_cnt++;
    }

    //TODO: more validations are possible.

    return err_cnt;
}

// Copy video input data to picture buffer
static void copy_picture_data(VmafPicture *pic, video_input_ycbcr ycbcr,
                               video_input_info *info, int depth)
{
    if (info->depth == depth) {
        if (info->depth == 8) {
            for (unsigned i = 0; i < 3; i++) {
                int xdec = i&&!(info->pixel_fmt&1);
                int ydec = i&&!(info->pixel_fmt&2);
                uint8_t *ycbcr_data = ycbcr[i].data +
                    (info->pic_y >> ydec) * ycbcr[i].stride +
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
                int xdec = i&&!(info->pixel_fmt&1);
                int ydec = i&&!(info->pixel_fmt&2);
                uint16_t *ycbcr_data = (uint16_t*) ycbcr[i].data +
                    (info->pic_y >> ydec) * (ycbcr[i].stride / 2) +
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
                int xdec = i&&!(info->pixel_fmt&1);
                int ydec = i&&!(info->pixel_fmt&2);
                uint8_t *ycbcr_data = ycbcr[i].data +
                    (info->pic_y >> ydec) * ycbcr[i].stride +
                    (info->pic_x >> xdec);
                uint16_t *pic_data = (uint16_t*)pic->data[i];

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
                int xdec = i&&!(info->pixel_fmt&1);
                int ydec = i&&!(info->pixel_fmt&2);
                uint16_t *ycbcr_data = (uint16_t*) ycbcr[i].data +
                    (info->pic_y >> ydec) * (ycbcr[i].stride / 2) +
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
    if (ret < 1) return !ret;

    video_input_get_info(vid, &info);

#ifdef VMAF_PICTURE_POOL
    ret = vmaf_fetch_preallocated_picture(vmaf, pic);
    if (ret) {
        fprintf(stderr, "problem fetching picture from pool.\n");
        return -1;
    }
#else
    (void) vmaf;  // Unused when pool is disabled
    ret = vmaf_picture_alloc(pic, pix_fmt_map(info.pixel_fmt), depth,
                             info.pic_w, info.pic_h);
    if (ret) {
        fprintf(stderr, "problem allocating picture.\n");
        return -1;
    }
#endif

    copy_picture_data(pic, ycbcr, &info, depth);
    return 0;
}

int main(int argc, char *argv[])
{
    int err = 0;
    const int istty = isatty(fileno(stderr));

    CLISettings c;
    cli_parse(argc, argv, &c);

    if (istty && !c.quiet) {
        fprintf(stderr, "VMAF version %s\n", vmaf_version());
    }

    FILE *file_ref = fopen(c.path_ref, "rb");
    if (!file_ref) {
        fprintf(stderr, "could not open file: %s\n", c.path_ref);
        return -1;
    }

    FILE *file_dist = fopen(c.path_dist, "rb");
    if (!file_dist) {
        fprintf(stderr, "could not open file: %s\n", c.path_dist);
        fclose(file_ref);
        return -1;
    }

    video_input vid_ref;
    if (c.use_yuv) {
        err = raw_input_open(&vid_ref, file_ref,
                             c.width, c.height, c.pix_fmt, c.bitdepth);
    } else {
        err = video_input_open(&vid_ref, file_ref);
    }
    if (err) {
        fprintf(stderr, "problem with reference file: %s\n", c.path_ref);
        fclose(file_ref);
        fclose(file_dist);
        return -1;
    }

    video_input vid_dist;
    if (c.use_yuv) {
        err = raw_input_open(&vid_dist, file_dist,
                             c.width, c.height, c.pix_fmt, c.bitdepth);
    } else {
        err = video_input_open(&vid_dist, file_dist);
    }
    if (err) {
        fprintf(stderr, "problem with distorted file: %s\n", c.path_dist);
        video_input_close(&vid_ref);
        fclose(file_dist);
        return -1;
    }

    err = validate_videos(&vid_ref, &vid_dist, c.common_bitdepth);
    if (err) {
        fprintf(stderr, "videos are incompatible, %d %s.\n",
                err, err == 1 ? "problem" : "problems");
        video_input_close(&vid_ref);
        video_input_close(&vid_dist);
        return -1;
    }

    int common_bitdepth;
    if (c.use_yuv) {
        common_bitdepth = c.bitdepth;
    } else {
        video_input_info info1, info2;
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

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, cfg);
    if (err) {
        fprintf(stderr, "problem initializing VMAF context\n");
        return -1;
    }

    // GPU backend initialization: each backend activates only when its
    // specific flag is passed.  --gpumask enables the preferred backend
    // (SYCL > CUDA).  --sycl_device selects
    // that specific backend.  No flag = CPU only.
#ifdef HAVE_SYCL
    bool sycl_active = false;
    VmafSyclState *sycl_state;
    VmafSyclConfiguration sycl_cfg = {
        .device_index = c.sycl_device >= 0 ? c.sycl_device : 0,
    };
    if ((c.sycl_device >= 0 || c.use_gpumask) && !c.no_sycl) {
        err = vmaf_sycl_state_init(&sycl_state, sycl_cfg);
        if (err) {
            fprintf(stderr, "problem during vmaf_sycl_state_init, using CPU\n");
        } else {
            err = vmaf_sycl_import_state(vmaf, sycl_state);
            if (err) {
                fprintf(stderr, "problem during vmaf_sycl_import_state\n");
                return -1;
            }
            sycl_active = true;
        }
    }
#endif
#ifdef HAVE_CUDA
    bool cuda_active = false;
    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = { 0 };
    if (c.use_gpumask && !c.no_cuda
#ifdef HAVE_SYCL
        && !sycl_active
#endif
    ) {
        err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
        if (err) {
            fprintf(stderr, "problem during vmaf_cuda_state_init, using CPU\n");
        } else {
            err |= vmaf_cuda_import_state(vmaf, cu_state);
            if (err) {
                fprintf(stderr, "problem during vmaf_cuda_import_state\n");
                return -1;
            }
            cuda_active = true;
        }
    }
#endif

#ifdef VMAF_PICTURE_POOL
    // Preallocate picture pool to avoid allocation overhead
    video_input_info info;
    video_input_get_info(&vid_ref, &info);

    VmafPictureConfiguration pic_cfg = {
        .pic_params = {
            .w = info.pic_w,
            .h = info.pic_h,
            .bpc = common_bitdepth,
            .pix_fmt = pix_fmt_map(info.pixel_fmt),
        },
        .pic_cnt = c.thread_cnt > 0 ? (c.thread_cnt + 1) * 2 : 2,
    };

    err = vmaf_preallocate_pictures(vmaf, pic_cfg);
    if (err) {
        fprintf(stderr, "problem during vmaf_preallocate_pictures\n");
        return -1;
    }

    if (istty && !c.quiet) {
        fprintf(stderr, "picture pool: %d pictures pre-allocated\n",
                pic_cfg.pic_cnt);
    }
#endif

    VmafModel **model;
    const size_t model_sz = sizeof(*model) * c.model_cnt;
    model = malloc(model_sz);
    memset(model, 0, model_sz);

    VmafModelCollection **model_collection;
    const size_t model_collection_sz =
        sizeof(*model_collection) * c.model_cnt;
    model_collection = malloc(model_sz);
    memset(model_collection, 0, model_collection_sz);

    const char *model_collection_label[c.model_cnt];
    unsigned model_collection_cnt = 0;

    for (unsigned i = 0; i < c.model_cnt; i++) {
        if (c.model_config[i].version) {
            err = vmaf_model_load(&model[i], &c.model_config[i].cfg,
                                  c.model_config[i].version);
        } else {
            err = vmaf_model_load_from_path(&model[i], &c.model_config[i].cfg,
                                            c.model_config[i].path);
        }

        if (err) {
            // check for model_collection before failing
            // this is implicit because the `--model` option could take either
            // a model or model_collection
            if (c.model_config[i].version) {
                err = vmaf_model_collection_load(&model[i],
                                        &model_collection[model_collection_cnt],
                                        &c.model_config[i].cfg,
                                        c.model_config[i].version);
            } else {
                err = vmaf_model_collection_load_from_path(&model[i],
                                        &model_collection[model_collection_cnt],
                                        &c.model_config[i].cfg,
                                        c.model_config[i].path);
            }

            if (err) {
                fprintf(stderr, "problem loading model: %s\n",
                        c.model_config[i].version ?
                            c.model_config[i].version : c.model_config[i].path);
                return -1;
            }

            model_collection_label[model_collection_cnt] =
                c.model_config[i].version ?
                    c.model_config[i].version : c.model_config[i].path;

            for (unsigned j = 0; j < c.model_config[i].overload_cnt; j++) {
                err = vmaf_model_collection_feature_overload(
                               model[i],
                               &model_collection[model_collection_cnt],
                               c.model_config[i].feature_overload[j].name,
                               c.model_config[i].feature_overload[j].opts_dict);
                if (err) {
                    fprintf(stderr,
                            "problem overloading feature extractors from "
                            "model collection: %s\n",
                            c.model_config[i].version ?
                            c.model_config[i].version : c.model_config[i].path);
                    return -1;
                }
            }

            err = vmaf_use_features_from_model_collection(vmaf,
                                        model_collection[model_collection_cnt]);
            if (err) {
                fprintf(stderr,
                        "problem loading feature extractors from "
                        "model collection: %s\n",
                        c.model_config[i].version ?
                            c.model_config[i].version : c.model_config[i].path);
                return -1;
            }

            model_collection_cnt++;
            continue;
        }

        for (unsigned j = 0; j < c.model_config[i].overload_cnt; j++) {
            err = vmaf_model_feature_overload(model[i],
                               c.model_config[i].feature_overload[j].name,
                               c.model_config[i].feature_overload[j].opts_dict);
            if (err) {
                fprintf(stderr,
                        "problem overloading feature extractors from "
                        "model: %s\n",
                        c.model_config[i].version ?
                            c.model_config[i].version : c.model_config[i].path);
                return -1;

            }
        }

        err = vmaf_use_features_from_model(vmaf, model[i]);
        if (err) {
            fprintf(stderr,
                    "problem loading feature extractors from model: %s\n",
                     c.model_config[i].version ?
                         c.model_config[i].version : c.model_config[i].path);
            return -1;
        }
    }

    for (unsigned i = 0; i < c.feature_cnt; i++) {
        err = vmaf_use_feature(vmaf, c.feature_cfg[i].name,
                               c.feature_cfg[i].opts_dict);
        if (err) {
            fprintf(stderr, "problem loading feature extractor: %s\n",
                    c.feature_cfg[i].name);
            return -1;
        }
    }

    if (c.tiny_model_path) {
        if (!vmaf_dnn_available()) {
            fprintf(stderr,
                    "--tiny-model requested (%s) but libvmaf was built "
                    "without DNN support (-Denable_dnn=disabled).\n",
                    c.tiny_model_path);
            return -1;
        }
        VmafDnnDevice dev = VMAF_DNN_DEVICE_AUTO;
        if (c.tiny_device) {
            if      (!strcmp(c.tiny_device, "cpu"))      dev = VMAF_DNN_DEVICE_CPU;
            else if (!strcmp(c.tiny_device, "cuda"))     dev = VMAF_DNN_DEVICE_CUDA;
            else if (!strcmp(c.tiny_device, "openvino")) dev = VMAF_DNN_DEVICE_OPENVINO;
            else if (!strcmp(c.tiny_device, "rocm"))     dev = VMAF_DNN_DEVICE_ROCM;
        }
        VmafDnnConfig dnn_cfg = {
            .device       = dev,
            .device_index = 0,
            .threads      = c.tiny_threads,
            .fp16_io      = c.tiny_fp16,
        };
        err = vmaf_use_tiny_model(vmaf, c.tiny_model_path, &dnn_cfg);
        if (err) {
            fprintf(stderr, "problem loading tiny model %s: %d\n",
                    c.tiny_model_path, err);
            return -1;
        }
    }

    VmafPicture pic_ref, pic_dist;

    for (unsigned i = 0; i < c.frame_skip_ref; i++)
        fetch_picture(vmaf, &vid_ref, &pic_ref, common_bitdepth);

    for (unsigned i = 0; i < c.frame_skip_dist; i++)
        fetch_picture(vmaf, &vid_dist, &pic_dist, common_bitdepth);

    float fps = 0.;
    const time_t t0 = clock();
    unsigned picture_index;
    for (picture_index = 0 ;; picture_index++) {

        if (c.frame_cnt && picture_index >= c.frame_cnt)
            break;

        VmafPicture pic_ref, pic_dist;
        int ret1 = fetch_picture(vmaf, &vid_ref, &pic_ref, common_bitdepth);
        int ret2 = fetch_picture(vmaf, &vid_dist, &pic_dist, common_bitdepth);

        if (ret1 && ret2) {
            break;
        } else if (ret1 < 0 || ret2 < 0) {
            fprintf(stderr, "\nproblem while reading pictures\n");
            break;
        } else if (ret1) {
            fprintf(stderr, "\n\"%s\" ended before \"%s\".\n",
                    c.path_ref, c.path_dist);
            int err = vmaf_picture_unref(&pic_dist);
            if (err)
                fprintf(stderr, "\nproblem during vmaf_picture_unref\n");
            break;
        } else if (ret2) {
            fprintf(stderr, "\n\"%s\" ended before \"%s\".\n",
                    c.path_dist, c.path_ref);
            int err = vmaf_picture_unref(&pic_ref);
            if (err)
                fprintf(stderr, "\nproblem during vmaf_picture_unref\n");
            break;
        }

        if (istty && !c.quiet) {
            if (picture_index > 0 && !(picture_index % 10)) {
                fps = (picture_index + 1) /
                      (((float)clock() - t0) / CLOCKS_PER_SEC);
            }

            fprintf(stderr, "\r%d frame%s %s %.2f FPS\033[K",
                    picture_index + 1, picture_index ? "s" : " ",
                    spinner[picture_index % spinner_length], fps);
            fflush(stderr);
        }

        err = vmaf_read_pictures(vmaf, &pic_ref, &pic_dist, picture_index);
        if (err) {
            fprintf(stderr, "\nproblem reading pictures\n");
            break;
        }
    }
    if (istty && !c.quiet)
        fprintf(stderr, "\n");

    err |= vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) {
        fprintf(stderr, "problem flushing context\n");
        return err;
    }

    if (!c.no_prediction) {
        for (unsigned i = 0; i < c.model_cnt; i++) {
            double vmaf_score;
            err = vmaf_score_pooled(vmaf, model[i], VMAF_POOL_METHOD_MEAN,
                                    &vmaf_score, 0, picture_index - 1);
            if (err) {
                fprintf(stderr, "problem generating pooled VMAF score\n");
                return -1;
            }

            if (istty && (!c.quiet || !c.output_path)) {
                fprintf(stderr, "%s: ",
                        c.model_config[i].version ?
                            c.model_config[i].version : c.model_config[i].path);
                fprintf(stderr, c.precision_fmt, vmaf_score);
                fprintf(stderr, "\n");
            }
        }

        for (unsigned i = 0; i < model_collection_cnt; i++) {
            VmafModelCollectionScore score = { 0 };
            err = vmaf_score_pooled_model_collection(vmaf, model_collection[i],
                                                     VMAF_POOL_METHOD_MEAN, &score,
                                                     0, picture_index - 1);
            if (err) {
                fprintf(stderr, "problem generating pooled VMAF score\n");
                return -1;
            }

            switch (score.type) {
            case VMAF_MODEL_COLLECTION_SCORE_BOOTSTRAP:
                if (istty && (!c.quiet || !c.output_path)) {
                    fprintf(stderr, "%s: ", model_collection_label[i]);
                    fprintf(stderr, c.precision_fmt, score.bootstrap.bagging_score);
                    fprintf(stderr, ", ci.p95: [");
                    fprintf(stderr, c.precision_fmt, score.bootstrap.ci.p95.lo);
                    fprintf(stderr, ", ");
                    fprintf(stderr, c.precision_fmt, score.bootstrap.ci.p95.hi);
                    fprintf(stderr, "], stddev: ");
                    fprintf(stderr, c.precision_fmt, score.bootstrap.stddev);
                    fprintf(stderr, "\n");
                }
                break;
            default:
                break;
            }
        }
    }

    if (c.output_path)
        vmaf_write_output_with_format(vmaf, c.output_path, c.output_fmt,
                                      c.precision_fmt);

    for (unsigned i = 0; i < c.model_cnt; i++)
        vmaf_model_destroy(model[i]);
    free(model);

    for (unsigned i = 0; i < model_collection_cnt; i++)
        vmaf_model_collection_destroy(model_collection[i]);
    free(model_collection);

    video_input_close(&vid_ref);
    video_input_close(&vid_dist);
    vmaf_close(vmaf);
#ifdef HAVE_SYCL
    if (sycl_active)
        vmaf_sycl_state_free(&sycl_state);
#endif
    cli_free(&c);
    return err;
}
