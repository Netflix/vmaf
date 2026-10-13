/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include <getopt.h>
#include <stdio.h>
#include <string.h>

#include "test.h"

#include "cli_parse.h"
#include "config.h"

#ifdef HAVE_GETOPT_H
#include <getopt.h>
#else
#error "Meson target is missing getopt_dependency"
#endif

static int cli_free_dicts(CLISettings *settings) {
    for (unsigned i = 0; i < settings->feature_cnt; i++) {
        int err = vmaf_feature_dictionary_free(&(settings->feature_cfg[i].opts_dict));
        if (err) return err;
    }
    return 0;
}

static char *test_aom_ctc_v1_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v1.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v1.0 provided but common_bitdepth enabled", !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v1.0 provided but number of features is not 5", settings.feature_cnt == 5);
    mu_assert("cli_parse: --aom_ctc v1.0 provided but number of models is not 2", settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v2_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v2.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v2.0 provided but common_bitdepth enabled", !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v2.0 provided but number of features is not 5", settings.feature_cnt == 5);
    mu_assert("cli_parse: --aom_ctc v2.0 provided but number of models is not 2", settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v3_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v3.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v3.0 provided but common_bitdepth enabled", !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v3.0 provided but number of features is not 6", settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v3.0 provided but number of models is not 2", settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v4_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v4.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v4.0 provided but common_bitdepth enabled", !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v4.0 provided but number of features is not 6", settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v4.0 provided but number of models is not 2", settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v5_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v5.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v5.0 provided but common_bitdepth enabled", !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v5.0 provided but number of features is not 6", settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v5.0 provided but number of models is not 2", settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_aom_ctc_v6_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--aom_ctc", "v6.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --aom_ctc v6.0 provided but common_bitdepth not enabled", settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v6.0 provided but number of features is not 6", settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v6.0 provided but number of models is not 2", settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_nflx_ctc_v1_0()
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--nflx_ctc", "v1.0"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --nflx_ctc v1.0 provided but common_bitdepth enabled", !settings.common_bitdepth);
    mu_assert("cli_parse: --nflx_ctc v1.0 provided but number of features is not 3", settings.feature_cnt == 3);
    mu_assert("cli_parse: --nflx_ctc v1.0 provided but number of models is not 2", settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_color_metadata_defaults_to_unknown()
{
    char *argv[13] = {"vmaf", "-r", "ref.yuv", "-d", "dis.yuv",
                      "-w", "16", "-h", "16", "-p", "420", "-b", "8"};
    int argc = 13;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: color.range should default to unknown",
              settings.color_ref.range == VMAF_COLOR_RANGE_UNKNOWN);
    mu_assert("cli_parse: color.primaries should default to unknown",
              settings.color_ref.primaries == VMAF_COLOR_PRIMARIES_UNKNOWN);
    mu_assert("cli_parse: color.trc should default to unknown",
              settings.color_ref.trc == VMAF_COLOR_TRC_UNKNOWN);
    mu_assert("cli_parse: color.matrix should default to unknown",
              settings.color_ref.matrix == VMAF_COLOR_MATRIX_UNKNOWN);
    mu_assert("cli_parse: distorted color should default to unknown",
              settings.color_dist.range == VMAF_COLOR_RANGE_UNKNOWN &&
              settings.color_dist.primaries == VMAF_COLOR_PRIMARIES_UNKNOWN &&
              settings.color_dist.trc == VMAF_COLOR_TRC_UNKNOWN &&
              settings.color_dist.matrix == VMAF_COLOR_MATRIX_UNKNOWN);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_color_metadata_parses()
{
    char *argv[29] = {"vmaf", "-r", "ref.yuv", "-d", "dis.yuv", "-w", "16", "-h", "16", "-p", "420", "-b", "8",
                      "--color_range_ref", "full",
                      "--color_range_dist", "full",
                      "--color_primaries_ref", "bt2020",
                      "--color_primaries_dist", "bt2020",
                      "--color_trc_ref", "smpte2084",
                      "--color_trc_dist", "smpte2084",
                      "--color_matrix_ref", "bt2020nc",
                      "--color_matrix_dist", "bt2020nc"};
    int argc = 29;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --color_range full not parsed",
              settings.color_ref.range == VMAF_COLOR_RANGE_FULL);
    mu_assert("cli_parse: --color_primaries bt2020 not parsed",
              settings.color_ref.primaries == VMAF_COLOR_PRIMARIES_BT2020);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_color_trc_and_matrix_parse()
{
    char *argv[29] = {"vmaf", "-r", "ref.yuv", "-d", "dis.yuv", "-w", "16", "-h", "16", "-p", "420", "-b", "10",
                      "--color_range_ref", "limited",
                      "--color_range_dist", "limited",
                      "--color_primaries_ref", "bt2020",
                      "--color_primaries_dist", "bt2020",
                      "--color_trc_ref", "pq",
                      "--color_trc_dist", "pq",
                      "--color_matrix_ref", "bt2020nc",
                      "--color_matrix_dist", "bt2020nc"};
    int argc = 29;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --color_trc pq not parsed",
              settings.color_ref.trc == VMAF_COLOR_TRC_SMPTE2084);
    mu_assert("cli_parse: --color_matrix bt2020nc not parsed",
              settings.color_ref.matrix == VMAF_COLOR_MATRIX_BT2020_NCL);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_color_trc_and_matrix_aliases_parse()
{
    char *argv[29] = {"vmaf", "-r", "ref.yuv", "-d", "dis.yuv", "-w", "16", "-h", "16", "-p", "420", "-b", "10",
                      "--color_range_ref", "limited",
                      "--color_range_dist", "limited",
                      "--color_primaries_ref", "bt709",
                      "--color_primaries_dist", "bt709",
                      "--color_trc_ref", "pq",
                      "--color_trc_dist", "pq",
                      "--color_matrix_ref", "ictcp",
                      "--color_matrix_dist", "ictcp"};
    int argc = 29;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --color_trc pq alias not parsed",
              settings.color_ref.trc == VMAF_COLOR_TRC_SMPTE2084);
    mu_assert("cli_parse: --color_matrix ictcp not parsed",
              settings.color_ref.matrix == VMAF_COLOR_MATRIX_ICTCP);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static bool color_is(const VmafColor *c, enum VmafColorRange range,
                     enum VmafColorPrimaries primaries,
                     enum VmafColorTransferCharacteristic trc,
                     enum VmafColorMatrixCoefficients matrix)
{
    return c->range == range && c->primaries == primaries &&
           c->trc == trc && c->matrix == matrix;
}

static char *test_per_side_color_flags_differ()
{
    char *argv[29] = {"vmaf", "-r", "ref.yuv", "-d", "dis.yuv",
                      "-w", "16", "-h", "16", "-p", "420", "-b", "10",
                      "--color_range_ref", "limited",
                      "--color_primaries_ref", "bt2020",
                      "--color_trc_ref", "pq",
                      "--color_matrix_ref", "bt2020nc",
                      "--color_range_dist", "full",
                      "--color_primaries_dist", "bt709",
                      "--color_trc_dist", "bt709",
                      "--color_matrix_dist", "bt709"};
    int argc = 29;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --color_*_ref should set only the reference",
              color_is(&settings.color_ref, VMAF_COLOR_RANGE_LIMITED,
                       VMAF_COLOR_PRIMARIES_BT2020, VMAF_COLOR_TRC_SMPTE2084,
                       VMAF_COLOR_MATRIX_BT2020_NCL));
    mu_assert("cli_parse: --color_*_dist should set only the distorted input",
              color_is(&settings.color_dist, VMAF_COLOR_RANGE_FULL,
                       VMAF_COLOR_PRIMARIES_BT709, VMAF_COLOR_TRC_BT709,
                       VMAF_COLOR_MATRIX_BT709));
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_one_input_can_be_left_unspecified()
{
    char *argv[21] = {"vmaf", "-r", "ref.yuv", "-d", "dis.yuv",
                      "-w", "16", "-h", "16", "-p", "420", "-b", "10",
                      "--color_range_ref", "limited",
                      "--color_primaries_ref", "bt2020",
                      "--color_trc_ref", "pq",
                      "--color_matrix_ref", "bt2020nc"};
    int argc = 21;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: the reference should be fully specified",
              color_is(&settings.color_ref, VMAF_COLOR_RANGE_LIMITED,
                       VMAF_COLOR_PRIMARIES_BT2020, VMAF_COLOR_TRC_SMPTE2084,
                       VMAF_COLOR_MATRIX_BT2020_NCL));
    mu_assert("cli_parse: the distorted input should stay unspecified",
              color_is(&settings.color_dist, VMAF_COLOR_RANGE_UNKNOWN,
                       VMAF_COLOR_PRIMARIES_UNKNOWN, VMAF_COLOR_TRC_UNKNOWN,
                       VMAF_COLOR_MATRIX_UNKNOWN));
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

#if VMAF_BUILT_IN_MODELS
static char *test_default_model_is_sdr_without_color_metadata()
{
    char *argv[13] = {"vmaf", "-r", "ref.yuv", "-d", "dis.yuv",
                      "-w", "16", "-h", "16", "-p", "420", "-b", "8"};
    int argc = 13;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: no model given should insert exactly one default",
              settings.model_cnt == 1);
    mu_assert("cli_parse: default model without color metadata should be "
              "the SDR default",
              !strcmp(settings.model_config[0].version, "vmaf_v0.6.1"));
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

static char *test_default_model_is_sdr_for_bt709_source()
{
    char *argv[29] = {"vmaf", "-r", "ref.yuv", "-d", "dis.yuv", "-w", "16", "-h", "16", "-p", "420", "-b", "8",
                      "--color_range_ref", "limited",
                      "--color_range_dist", "limited",
                      "--color_primaries_ref", "bt709",
                      "--color_primaries_dist", "bt709",
                      "--color_trc_ref", "bt709",
                      "--color_trc_dist", "bt709",
                      "--color_matrix_ref", "bt709",
                      "--color_matrix_dist", "bt709"};
    int argc = 29;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: default model for a fully-specified bt709 source "
              "should still be the SDR default",
              !strcmp(settings.model_config[0].version, "vmaf_v0.6.1"));
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

#endif


static void parse_model_option(CLISettings *settings, const char *spec)
{
    char *argv[7] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--model",
                     (char *) spec};
    optind = 1;
    cli_parse(7, argv, settings);
}

static void free_model_option(CLISettings *settings)
{
    for (unsigned i = 0; i < settings->model_cnt; i++) {
        CLIModelConfig *m = &settings->model_config[i];
        for (unsigned j = 0; j < m->overload_cnt; j++)
            vmaf_feature_dictionary_free(&m->feature_overload[j].opts_dict);
    }
    cli_free(settings);
}

static char *test_model_path_drive_letter()
{
    const char *const paths[] = {
        "C:\\models\\vmaf_v0.6.1.json",
        "C:/models/vmaf_v0.6.1.json",
        "d:\\vmaf.json",
        "Z:/a/b.json",
    };
    for (unsigned i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        char spec[128];
        snprintf(spec, sizeof(spec), "path=%s", paths[i]);
        CLISettings settings;
        parse_model_option(&settings, spec);
        mu_assert("cli_parse: --model drive letter path: model count is not 1",
                  settings.model_cnt == 1);
        mu_assert("cli_parse: --model drive letter path was split at the colon",
                  settings.model_config[0].path &&
                  !strcmp(settings.model_config[0].path, paths[i]));
        mu_assert("cli_parse: --model drive letter path created an overload",
                  settings.model_config[0].overload_cnt == 0);
        free_model_option(&settings);
    }
    return NULL;
}

static char *test_model_path_drive_letter_with_options()
{
    CLISettings settings;
    parse_model_option(&settings,
                       "path=C:\\models\\vmaf.json:name=mine:disable_clip");
    mu_assert("cli_parse: --model path then options: model count is not 1",
              settings.model_cnt == 1);
    CLIModelConfig *m = &settings.model_config[0];
    mu_assert("cli_parse: --model path then options: wrong path",
              m->path && !strcmp(m->path, "C:\\models\\vmaf.json"));
    mu_assert("cli_parse: --model path then options: wrong name",
              !strcmp(m->cfg.name, "mine"));
    mu_assert("cli_parse: --model path then options: disable_clip not set",
              m->cfg.flags & VMAF_MODEL_FLAG_DISABLE_CLIP);
    free_model_option(&settings);

    parse_model_option(&settings,
                       "name=mine:path=C:/models/vmaf.json:enable_transform");
    m = &settings.model_config[0];
    mu_assert("cli_parse: --model options around path: wrong path",
              m->path && !strcmp(m->path, "C:/models/vmaf.json"));
    mu_assert("cli_parse: --model options around path: wrong name",
              !strcmp(m->cfg.name, "mine"));
    mu_assert("cli_parse: --model options around path: enable_transform not set",
              m->cfg.flags & VMAF_MODEL_FLAG_ENABLE_TRANSFORM);
    free_model_option(&settings);

    parse_model_option(&settings,
                       "path=C:\\m.json:vif.vif_enhn_gain_limit=1.5");
    m = &settings.model_config[0];
    mu_assert("cli_parse: --model path then overload: wrong path",
              m->path && !strcmp(m->path, "C:\\m.json"));
    mu_assert("cli_parse: --model path then overload: overload count is not 1",
              m->overload_cnt == 1);
    mu_assert("cli_parse: --model path then overload: wrong feature name",
              !strcmp(m->feature_overload[0].name, "vif"));
    free_model_option(&settings);

    return NULL;
}

/* A single letter before a colon is only a drive letter when a path separator
 * follows the colon. Option keys never start with one, so a one-letter value
 * followed by the next option still splits. */
static char *test_model_one_letter_value_still_splits()
{
    CLISettings settings;
    parse_model_option(&settings, "name=a:path=/tmp/vmaf.json");
    mu_assert("cli_parse: --model one-letter name: model count is not 1",
              settings.model_cnt == 1);
    CLIModelConfig *m = &settings.model_config[0];
    mu_assert("cli_parse: --model one-letter name was merged with the next option",
              !strcmp(m->cfg.name, "a"));
    mu_assert("cli_parse: --model path after one-letter name is wrong",
              m->path && !strcmp(m->path, "/tmp/vmaf.json"));
    free_model_option(&settings);

    parse_model_option(&settings, "path=m:name=b:disable_clip");
    m = &settings.model_config[0];
    mu_assert("cli_parse: --model one-letter relative path was merged",
              m->path && !strcmp(m->path, "m"));
    mu_assert("cli_parse: --model name after one-letter path is wrong",
              !strcmp(m->cfg.name, "b"));
    mu_assert("cli_parse: --model disable_clip after one-letter path not set",
              m->cfg.flags & VMAF_MODEL_FLAG_DISABLE_CLIP);
    free_model_option(&settings);

    return NULL;
}

static char *test_model_existing_strings_unchanged()
{
    CLISettings settings;
    parse_model_option(&settings,
                       "version=vmaf_v0.6.1:name=vmaf_x:disable_clip:enable_transform");
    CLIModelConfig *m = &settings.model_config[0];
    mu_assert("cli_parse: --model version: wrong version",
              m->version && !strcmp(m->version, "vmaf_v0.6.1"));
    mu_assert("cli_parse: --model version: wrong name",
              !strcmp(m->cfg.name, "vmaf_x"));
    mu_assert("cli_parse: --model version: flags not set",
              (m->cfg.flags & VMAF_MODEL_FLAG_DISABLE_CLIP) &&
              (m->cfg.flags & VMAF_MODEL_FLAG_ENABLE_TRANSFORM));
    mu_assert("cli_parse: --model version: path should be unset", !m->path);
    free_model_option(&settings);

    parse_model_option(&settings,
                       "path=model/vmaf_v0.6.1.json:name=rel");
    m = &settings.model_config[0];
    mu_assert("cli_parse: --model relative path: wrong path",
              m->path && !strcmp(m->path, "model/vmaf_v0.6.1.json"));
    mu_assert("cli_parse: --model relative path: wrong name",
              !strcmp(m->cfg.name, "rel"));
    free_model_option(&settings);

    parse_model_option(&settings, "path=/opt/vmaf/model.json");
    m = &settings.model_config[0];
    mu_assert("cli_parse: --model absolute path: wrong path",
              m->path && !strcmp(m->path, "/opt/vmaf/model.json"));
    free_model_option(&settings);

    return NULL;
}


char *run_tests()
{
    mu_run_test(test_aom_ctc_v1_0);
    mu_run_test(test_aom_ctc_v2_0);
    mu_run_test(test_aom_ctc_v3_0);
    mu_run_test(test_aom_ctc_v4_0);
    mu_run_test(test_aom_ctc_v5_0);
    mu_run_test(test_aom_ctc_v6_0);
    mu_run_test(test_nflx_ctc_v1_0);
    mu_run_test(test_color_metadata_defaults_to_unknown);
    mu_run_test(test_color_metadata_parses);
    mu_run_test(test_color_trc_and_matrix_parse);
    mu_run_test(test_color_trc_and_matrix_aliases_parse);
    mu_run_test(test_per_side_color_flags_differ);
    mu_run_test(test_one_input_can_be_left_unspecified);
#if VMAF_BUILT_IN_MODELS
    mu_run_test(test_default_model_is_sdr_without_color_metadata);
    mu_run_test(test_default_model_is_sdr_for_bt709_source);
#endif
    mu_run_test(test_model_path_drive_letter);
    mu_run_test(test_model_path_drive_letter_with_options);
    mu_run_test(test_model_one_letter_value_still_splits);
    mu_run_test(test_model_existing_strings_unchanged);
    return NULL;
}
