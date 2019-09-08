/**
 *
 *  Copyright 2016-2019 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#ifndef LIBVMAF_H_
#define LIBVMAF_H_

#define MAX_NUM_VMAF_MODELS 5 // there can be MAX_NUM_VMAF_MODELS - 1 additional models at most

#ifndef WINCE
#define TIME_TEST_ENABLE 		1 // 1: memory leak test enable 0: disable
#define MEM_LEAK_TEST_ENABLE 	0 // prints execution time in xml log when enabled.
#else
//For Windows memory leak test and execution time test cases are not handled.
#define TIME_TEST_ENABLE 0
#define MEM_LEAK_TEST_ENABLE 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

enum VmafLogFmt {
    VMAF_LOG_FMT_XML                                   = 0,
    VMAF_LOG_FMT_JSON                                  = 1,
    VMAF_LOG_FMT_CSV                                   = 2,
};

enum VmafPoolingMethod {
    VMAF_POOL_MIN                                      = 0,
    VMAF_POOL_MEAN                                     = 1,
    VMAF_POOL_HARMONIC_MEAN                            = 2,
};

typedef struct VmafFeatureCalculationSetting
{
    unsigned int n_threads;
    unsigned int n_subsample;
    bool disable_avx;
} VmafFeatureCalculationSetting;

enum VmafFeatureModeSetting {
    VMAF_FEATURE_MODE_SETTING_DO_NONE                  = (1 << 0),
    VMAF_FEATURE_MODE_SETTING_DO_PSNR                  = (1 << 1),
    VMAF_FEATURE_MODE_SETTING_DO_SSIM                  = (1 << 2),
    VMAF_FEATURE_MODE_SETTING_DO_MS_SSIM               = (1 << 3),
    VMAF_FEATURE_MODE_SETTING_DO_COLOR                 = (1 << 4),
};

enum VmafPixelFormat {
    VMAF_PIX_FMT_YUV420P                               = 0,
    VMAF_PIX_FMT_YUV422P                               = 1,
    VMAF_PIX_FMT_YUV444P                               = 2,
    VMAF_PIX_FMT_YUV420P10LE                           = 3,
    VMAF_PIX_FMT_YUV422P10LE                           = 4,
    VMAF_PIX_FMT_YUV444P10LE                           = 5,
    VMAF_PIX_FMT_UNKNOWN                               = 6,
};

typedef struct VmafPicture
{
    float *data[3];
    unsigned int w[3], h[3];
    unsigned int stride_byte[3];
    enum VmafPixelFormat pix_fmt;
} VmafPicture;

enum VmafModelSetting {
    VMAF_MODEL_SETTING_NONE                            = (1 << 0),
    VMAF_MODEL_SETTING_ENABLE_TRANSFORM                = (1 << 1),
    VMAF_MODEL_SETTING_DISABLE_CLIP                    = (1 << 2),
    VMAF_MODEL_SETTING_ENABLE_CONF_INTERVAL            = (1 << 3),
};

typedef struct {
    char *name;
    char *path;
    int vmaf_model_setting;
} VmafModel;

typedef struct {

    unsigned int width;
    unsigned int height;
    unsigned int num_models;
    unsigned int default_model_ind; // useful to tell between the default model and additional models (if any)

    char *log_path;
    int vmaf_feature_mode_setting;

    VmafModel vmaf_model[MAX_NUM_VMAF_MODELS];
    VmafFeatureCalculationSetting vmaf_feature_calculation_setting;

    enum VmafPixelFormat pix_fmt;
    enum VmafLogFmt log_fmt;
    enum VmafPoolingMethod pool_method;

} VmafSettings;

int compute_vmaf(double* vmaf_score,
                 int (*read_vmaf_picture)(VmafPicture *ref_vmaf_pict, VmafPicture *dis_vmaf_pict, float *temp_data, void *user_data),
				 void *user_data, VmafSettings *vmafSettings);

unsigned int get_additional_models(char *additional_model_paths, VmafModel *vmaf_model);

#ifdef __cplusplus
}
#endif

#endif /* _LIBVMAF_H */
