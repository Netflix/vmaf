#ifndef __VMAF_CLI_PARSE_H__
#define __VMAF_CLI_PARSE_H__

#include <stdbool.h>

#include "libvmaf/libvmaf.rc.h"

#define CLI_SETTINGS_STATIC_ARRAY_LEN 256

typedef struct {
    char *y4m_path_ref, *y4m_path_dist;
    char *log_path;
    char *model_path[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned model_cnt;
    char *feature[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned feature_cnt;
    char *import_path[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned import_cnt;
    enum VmafLogLevel log_level;
    unsigned thread_cnt;
    bool no_prediction;
} CLISettings;

void cli_parse(const int argc, char *const *const argv,
               CLISettings *const settings);

#endif /* __VMAF_CLI_PARSE_H__ */
