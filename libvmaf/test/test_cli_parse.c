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
    return NULL;
}
