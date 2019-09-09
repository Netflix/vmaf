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

#define MAX_NUM_VMAF_MODELS 21 // there can be MAX_NUM_VMAF_MODELS - 1 additional models at most

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

/** Thin wrapper for VMAF outputs (besides the logged features and predictions).
  * Can be extended to include other fields.
  **/
typedef struct VmafOutput
{
    double vmaf_score;
} VmafOutput;

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

/** Enum with settings for feature extraction.
  * For example, setting read_color to true will instruct read_vmaf_picture to read U and V planes.
  **/
typedef struct VmafFeatureCalculationSetting
{
    unsigned int n_threads;
    unsigned int n_subsample;
    bool disable_avx;
} VmafFeatureCalculationSetting;

/** Enum with settings for feature mode calculation.
  * If the setting is used, the corresponding feature calculation/operation will occur.
  * For example, the DO_PSNR setting will instruct extract PSNR.
  * The DO_CHROMA_PSNR setting will extract chroma PSNR scores.
  **/
enum VmafFeatureModeSetting {
    VMAF_FEATURE_MODE_SETTING_DO_NONE                  = (1 << 0),
    VMAF_FEATURE_MODE_SETTING_DO_PSNR                  = (1 << 1),
    VMAF_FEATURE_MODE_SETTING_DO_SSIM                  = (1 << 2),
    VMAF_FEATURE_MODE_SETTING_DO_MS_SSIM               = (1 << 3),
    VMAF_FEATURE_MODE_SETTING_DO_CHROMA_PSNR           = (1 << 4),
};

/** Definition of pixel formats in the VMAF library.
  **/
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

/** Model specific settings.
  * The "default" model is the one in by model_path and with
  * additional options, e.g. disable_clip, enable_transform
  * and enable confidence interval.
  * To follow the convention this model should be named "vmaf".
  * Additional models can be named as needed and customized
  * independently using the desired model setting.
  */
enum VmafModelSetting {
    VMAF_MODEL_SETTING_NONE                            = (1 << 0),
    VMAF_MODEL_SETTING_ENABLE_TRANSFORM                = (1 << 1),
    VMAF_MODEL_SETTING_DISABLE_CLIP                    = (1 << 2),
    VMAF_MODEL_SETTING_ENABLE_CONF_INTERVAL            = (1 << 3),
};

/** Thin VMAF model wrapper.
  * Assumes the existence of the model in a filesystem path.
  * Includes a model setting for disabling clip, score transformation etc. */
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

/** Computes VMAF using a read_vmaf_picture callback,
  * a user-defined struct and a settings struct.
  * A single VMAF value is returned and all calculated
  * features and additional predictions are written into a log file. */
int compute_vmaf(VmafOutput *vmaf_output_ptr,
                 int (*read_vmaf_picture)(VmafPicture *ref_vmaf_pict, VmafPicture *dis_vmaf_pict, float *temp_data, void *user_data),
				 void *user_data, VmafSettings *vmafSettings);

/** Populates additional models to be used for prediction.
  * A maximum of
  * The function returns the number of additional models as an integer (0 for a NULL ptr).
  * The first model is the model corresponding to model_path.
  * The additional models are passed in using the additional_model_paths string.
  * This string contains model-specific parameters for each additional model (path, disable_clip, etc.)
  * This string should be formatted as an escaped json-like string with the following format:
  * {\"<name>\":{\"model_path\"\:\"<path>\"}}, e.g.:
  * {\"another_model\":{\"model_path\"\:\"some/path/to/some/model/another_model.pkl\"}}
  * The fields <name> and <path> are required and optional parameters can be used by adding an escaped comma, e.g.:
  * {\"<name>\":{\"model_path\"\:\"<path>\"\,\"disable_clip\"\:\"1\"}} --> disable clipping of predictions
  * {\"<name>\":{\"model_path\"\:\"<path>\"\,\"enable_transform\"\:\"1\"}} --> use transformation for the predictions (e.g. phone model transformation)
  * {\"<name>\":{\"model_path\"\:\"<path>\"\,\"enable_conf_interval\"\:\"1\"}} --> calculate confidence interval (CI) for the predictions (the model should support CIs)
  * {\"<name>\":{\"model_path\"\:\"<path>\"\,\"disable_clip\"\:\"0\"\,\"enable_transform\"\:\"1\"}} --> enable clipping and transform scores
  * Real example: {\"vmaf_061_CI_no_phone\"\:{\"model_path\"\:\"model/vmaf_b_v0.6.3/vmaf_b_v0.6.3.pkl\"\,\"enable_conf_interval\"\:\"1\"\,\"enable_transform\"\:\"0\"}}
  * The order of the optional parameters does not matter
  * If optional parameters are not populated, a default value of false is used.
  * */
unsigned int get_additional_models(char *additional_model_paths, VmafModel *vmaf_model);

#ifdef __cplusplus
}
#endif

#endif /* _LIBVMAF_H */
