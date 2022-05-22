#include <stdlib.h>
#include <string.h>

#include "feature/alias.h"
#include "log.h"
#include "libvmaf/libvmaf.h"
#include "model.h"

static enum VmafOutputFormat log_fmt_map(const char *log_fmt)
{
    if (log_fmt) {
        if (!strcmp(log_fmt, "xml"))
            return VMAF_OUTPUT_FORMAT_XML;
        if (!strcmp(log_fmt, "json"))
            return VMAF_OUTPUT_FORMAT_JSON;
        if (!strcmp(log_fmt, "csv"))
            return VMAF_OUTPUT_FORMAT_CSV;
        if (!strcmp(log_fmt, "sub"))
            return VMAF_OUTPUT_FORMAT_SUB;
    }

    return VMAF_OUTPUT_FORMAT_NONE;
}

static enum VmafPoolingMethod pool_method_map(const char *pool_method)
{
    if (pool_method) {
        if (!strcmp(pool_method, "min"))
            return VMAF_POOL_METHOD_MIN;
        if (!strcmp(pool_method, "mean"))
            return VMAF_POOL_METHOD_MEAN;
        if (!strcmp(pool_method, "harmonic_mean"))
            return VMAF_POOL_METHOD_HARMONIC_MEAN;
    }

    return VMAF_POOL_METHOD_MEAN;
}

static int pix_fmt_map(char *fmt)
{
    if (fmt) {
        if (!strcmp(fmt, "yuv420p"))
            return VMAF_PIX_FMT_YUV420P;
        if (!strcmp(fmt, "yuv422p"))
            return VMAF_PIX_FMT_YUV422P;
        if (!strcmp(fmt, "yuv444p"))
            return VMAF_PIX_FMT_YUV444P;
        if (!strcmp(fmt, "yuv420p10le"))
            return VMAF_PIX_FMT_YUV420P;
        if (!strcmp(fmt, "yuv420p12le"))
            return VMAF_PIX_FMT_YUV420P;
        if (!strcmp(fmt, "yuv420p16le"))
            return VMAF_PIX_FMT_YUV420P;
        if (!strcmp(fmt, "yuv422p10le"))
            return VMAF_PIX_FMT_YUV422P;
        if (!strcmp(fmt, "yuv444p10le"))
            return VMAF_PIX_FMT_YUV444P;
    }

    return VMAF_PIX_FMT_UNKNOWN;

}

static int bitdepth_map(char *fmt)
{
    if (!strcmp(fmt, "yuv420p10le"))
        return 10;
    if (!strcmp(fmt, "yuv422p10le"))
        return 10;
    if (!strcmp(fmt, "yuv444p10le"))
        return 10;
    if (!strcmp(fmt, "yuv420p12le"))
        return 12;
    if (!strcmp(fmt, "yuv420p16le"))
        return 16;

    return 8;
}

static void copy_data(float *src, VmafPicture *dst, unsigned width,
                      unsigned height, int src_stride)
{
    float *a = src;
    uint8_t *b = dst->data[0];
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            b[j] = a[j];
        }
        a += src_stride / sizeof(float);
        b += dst->stride[0];
    }
}

static void copy_data_hbd(float *src, VmafPicture *dst, unsigned width,
                          unsigned height, int src_stride, unsigned bpc)
{
    float *a = src;
    uint16_t *b = dst->data[0];
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            b[j] = a[j] * (1 << (bpc - 8));
        }
        a += src_stride / sizeof(float);
        b += dst->stride[0] / sizeof(uint16_t);
    }
}

int compute_vmaf(double* vmaf_score, char* fmt, int width, int height,
                 int (*read_frame)(float *ref_data, float *main_data,
                                   float *temp_data, int stride_byte,
                                   void *user_data),
                 void *user_data, char *model_path, char *log_path,
                 char *log_fmt, int disable_clip, int disable_avx,
                 int enable_transform, int phone_model, int do_psnr,
                 int do_ssim, int do_ms_ssim, char *pool_method,
                 int n_thread, int n_subsample, int enable_conf_interval)
{

    vmaf_set_log_level(VMAF_LOG_LEVEL_INFO);
    vmaf_log(VMAF_LOG_LEVEL_INFO, "`compute_vmaf()` is deprecated "
             "and will be removed in a future libvmaf version\n");

    int err = 0;

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = n_thread,
        .n_subsample = n_subsample,
        .cpumask = disable_avx ? -1 : 0,
    };

    VmafContext *vmaf;
    err = vmaf_init(&vmaf, cfg);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "problem initializing VMAF context\n");
        return -1;
    }

    enum VmafModelFlags flags = VMAF_MODEL_FLAGS_DEFAULT;
    if (disable_clip)
        flags |= VMAF_MODEL_FLAG_DISABLE_CLIP;
    if (enable_transform || phone_model)
        flags |= VMAF_MODEL_FLAG_ENABLE_TRANSFORM;

    VmafModelConfig model_cfg = {
        .name = "vmaf",
        .flags = flags,
    };

    VmafModel *model = NULL;
    VmafModelCollection *model_collection = NULL;

    if (enable_conf_interval) {
        err = vmaf_model_collection_load_from_path(&model, &model_collection,
                                                   &model_cfg, model_path);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem loading model file: %s\n", model_path);
            goto end;
        }
        err = vmaf_use_features_from_model_collection(vmaf, model_collection);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                    "problem loading feature extractors from model file: %s\n",
                    model_path);
            goto end;
        }
    } else {
        err = vmaf_model_load_from_path(&model, &model_cfg, model_path);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem loading model file: %s\n", model_path);
            goto end;
        }
        err = vmaf_use_features_from_model(vmaf, model);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                    "problem loading feature extractors from model file: %s\n",
                    model_path);
            goto end;
        }
    }

    if (do_psnr) {
        VmafFeatureDictionary *d = NULL;
        vmaf_feature_dictionary_set(&d, "enable_chroma", "false");

        err = vmaf_use_feature(vmaf, "psnr", d);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem loading feature extractor: psnr\n");
            goto end;
        }
    }

    if (do_ssim) {
        err = vmaf_use_feature(vmaf, "float_ssim", NULL);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem loading feature extractor: ssim\n");
            goto end;
        }
    }

    if (do_ms_ssim) {
        err = vmaf_use_feature(vmaf, "float_ms_ssim", NULL);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem loading feature extractor: ms_ssim\n");
            goto end;
        }
    }

    int stride = width * sizeof(float);
    float *ref_data = malloc(height * stride);
    float *main_data = malloc(height * stride);
    float *temp_data = malloc(height * stride);
    if (!ref_data | !main_data | !temp_data) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "problem allocating picture memory\n");
        err = -1;
        goto free_data;
    }

    unsigned picture_index;
    for (picture_index = 0 ;; picture_index++) {
        err = read_frame(ref_data, main_data, temp_data, stride, user_data);
        if (err == 1) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "problem during read_frame\n");
            goto free_data;
        } else if (err == 2) {
            break; //EOF
        }

        VmafPicture pic_ref, pic_dist;
        err = vmaf_picture_alloc(&pic_ref, pix_fmt_map(fmt),
                                 bitdepth_map(fmt), width, height);
        err |= vmaf_picture_alloc(&pic_dist, pix_fmt_map(fmt),
                                  bitdepth_map(fmt), width, height);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem allocating picture memory\n");
            vmaf_picture_unref(&pic_ref);
            vmaf_picture_unref(&pic_dist);
            goto free_data;
        }

        const unsigned bpc = bitdepth_map(fmt);
        if (bpc > 8) {
            copy_data_hbd(ref_data, &pic_ref, width, height, stride, bpc);
            copy_data_hbd(main_data, &pic_dist, width, height, stride, bpc);
        } else {
            copy_data(ref_data, &pic_ref, width, height, stride);
            copy_data(main_data, &pic_dist, width, height, stride);
        }

        err = vmaf_read_pictures(vmaf, &pic_ref, &pic_dist, picture_index);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "problem reading pictures\n");
            break;
        }
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "problem flushing context\n");
        return err;
    }

     if (enable_conf_interval) {
         VmafModelCollectionScore model_collection_score;
         err = vmaf_score_pooled_model_collection(vmaf, model_collection,
                                                  pool_method_map(pool_method),
                                                  &model_collection_score, 0,
                                                  picture_index - 1);
         if (err) {
             vmaf_log(VMAF_LOG_LEVEL_ERROR,
                      "problem generating pooled VMAF score\n");
             goto free_data;
         }
    }

     err = vmaf_score_pooled(vmaf, model, pool_method_map(pool_method),
                             vmaf_score, 0, picture_index - 1);
     if (err) {
         vmaf_log(VMAF_LOG_LEVEL_ERROR,
                  "problem generating pooled VMAF score\n");
         goto free_data;
     }

    enum VmafOutputFormat output_fmt = log_fmt_map(log_fmt);
    if (output_fmt == VMAF_OUTPUT_FORMAT_NONE && log_path) {
        output_fmt = VMAF_OUTPUT_FORMAT_XML;
        vmaf_log(VMAF_LOG_LEVEL_WARNING, "use default log_fmt xml");
    }
    if (output_fmt) {
        vmaf_use_vmafossexec_aliases();
        err = vmaf_write_output(vmaf, log_path, output_fmt);
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "could not write output: %s\n", log_path);
            goto free_data;
        }
    }

free_data:
    if (ref_data) free(ref_data);
    if (main_data) free(main_data);
    if (temp_data) free(temp_data);
end:
    vmaf_model_destroy(model);
    vmaf_model_collection_destroy(model_collection);
    vmaf_close(vmaf);
    return err;
}
