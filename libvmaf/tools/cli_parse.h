#ifndef __VMAF_CLI_PARSE_H__
#define __VMAF_CLI_PARSE_H__

#include <stdbool.h>
#include <stdint.h>

#include "libvmaf/libvmaf.rc.h"
#include "libvmaf/model.h"
#include "libvmaf/feature.h"

#define CLI_SETTINGS_STATIC_ARRAY_LEN 256

typedef struct {
    const char *name;
    VmafFeatureDictionary *opts_dict;
} CLIFeatureConfig;

typedef struct {
    char *path_ref, *path_dist;
    unsigned width, height;
    enum VmafPixelFormat pix_fmt;
    unsigned bitdepth;
    bool use_yuv;
    char *output_path;
    enum VmafOutputFormat output_fmt;
    VmafModelConfig model_config[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned model_cnt;
    CLIFeatureConfig feature_cfg[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned feature_cnt;
    enum VmafLogLevel log_level;
    unsigned subsample;
    unsigned thread_cnt;
    bool no_prediction;
    uint32_t cpumask;
} CLISettings;

void cli_parse(const int argc, char *const *const argv,
               CLISettings *const settings);

#endif /* __VMAF_CLI_PARSE_H__ */
