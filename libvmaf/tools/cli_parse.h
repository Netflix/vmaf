#ifndef __VMAF_CLI_PARSE_H__
#define __VMAF_CLI_PARSE_H__

#include <stdbool.h>
#include <stdint.h>

#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"
#include "libvmaf/feature.h"

#define CLI_SETTINGS_STATIC_ARRAY_LEN 32

typedef struct {
    const char *name;
    VmafFeatureDictionary *opts_dict;
    void *buf;
} CLIFeatureConfig;

typedef struct {
    const char *path;
    const char *version;
    VmafModelConfig cfg;
    struct {
        const char *name;
        VmafFeatureDictionary *opts_dict;
    } feature_overload[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned overload_cnt;
    void *buf;
} CLIModelConfig;

typedef struct {
    char *path_ref, *path_dist;
    unsigned frame_skip_ref;
    unsigned frame_skip_dist;
    unsigned frame_cnt;
    unsigned width, height;
    enum VmafPixelFormat pix_fmt;
    unsigned bitdepth;
    bool use_yuv;
    char *output_path;
    enum VmafOutputFormat output_fmt;
    CLIModelConfig model_config[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned model_cnt;
    CLIFeatureConfig feature_cfg[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned feature_cnt;
    enum VmafLogLevel log_level;
    unsigned subsample;
    unsigned thread_cnt;
    bool no_prediction;
    bool quiet;
    bool common_bitdepth;
    unsigned cpumask;
    unsigned gpumask;
} CLISettings;

void cli_parse(const int argc, char *const *const argv,
               CLISettings *const settings);

void cli_free(CLISettings *settings);

#endif /* __VMAF_CLI_PARSE_H__ */
