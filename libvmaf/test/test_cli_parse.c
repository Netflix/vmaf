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

#ifdef _WIN32
#include "compat/win32/getopt.h"
#else
#include <getopt.h>
#endif

#include "test.h"

#include "cli_parse.h"

static int cli_free_dicts(CLISettings *settings)
{
    for (unsigned i = 0; i < settings->feature_cnt; i++) {
        int err = vmaf_feature_dictionary_free(&(settings->feature_cfg[i].opts_dict));
        if (err)
            return err;
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
    mu_assert("cli_parse: --aom_ctc v1.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v1.0 provided but number of features is not 5",
              settings.feature_cnt == 5);
    mu_assert("cli_parse: --aom_ctc v1.0 provided but number of models is not 2",
              settings.model_cnt == 2);
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
    mu_assert("cli_parse: --aom_ctc v2.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v2.0 provided but number of features is not 5",
              settings.feature_cnt == 5);
    mu_assert("cli_parse: --aom_ctc v2.0 provided but number of models is not 2",
              settings.model_cnt == 2);
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
    mu_assert("cli_parse: --aom_ctc v3.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v3.0 provided but number of features is not 6",
              settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v3.0 provided but number of models is not 2",
              settings.model_cnt == 2);
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
    mu_assert("cli_parse: --aom_ctc v4.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v4.0 provided but number of features is not 6",
              settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v4.0 provided but number of models is not 2",
              settings.model_cnt == 2);
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
    mu_assert("cli_parse: --aom_ctc v5.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v5.0 provided but number of features is not 6",
              settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v5.0 provided but number of models is not 2",
              settings.model_cnt == 2);
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
    mu_assert("cli_parse: --aom_ctc v6.0 provided but common_bitdepth not enabled",
              settings.common_bitdepth);
    mu_assert("cli_parse: --aom_ctc v6.0 provided but number of features is not 6",
              settings.feature_cnt == 6);
    mu_assert("cli_parse: --aom_ctc v6.0 provided but number of models is not 2",
              settings.model_cnt == 2);
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
    mu_assert("cli_parse: --nflx_ctc v1.0 provided but common_bitdepth enabled",
              !settings.common_bitdepth);
    mu_assert("cli_parse: --nflx_ctc v1.0 provided but number of features is not 3",
              settings.feature_cnt == 3);
    mu_assert("cli_parse: --nflx_ctc v1.0 provided but number of models is not 2",
              settings.model_cnt == 2);
    cli_free(&settings);
    cli_free_dicts(&settings);

    return NULL;
}

/* `--backend cuda` must end up with gpumask == 0 (NOT 1), because
 * VmafConfiguration::gpumask is a CUDA-*disable* bitmask — any nonzero
 * value disables CUDA in compute_fex_flags. The CLI's job is only to
 * trip use_gpumask so vmaf_cuda_state_init runs; the runtime then
 * picks the CUDA extractors because gpumask is 0. Earlier revisions
 * set gpumask = 1 here, which silently routed every "CUDA" run
 * through the CPU path. */
static char *test_backend_cuda_engages_cuda()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "cuda"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend cuda must set use_gpumask = true (so CUDA inits)",
              settings.use_gpumask);
    mu_assert("cli_parse: --backend cuda must set gpumask = 0 (any nonzero DISABLES CUDA)",
              settings.gpumask == 0);
    mu_assert("cli_parse: --backend cuda must set no_sycl = true", settings.no_sycl);
    mu_assert("cli_parse: --backend cuda must set no_vulkan = true", settings.no_vulkan);
    mu_assert("cli_parse: --backend cuda must set no_hip = true", settings.no_hip);
    mu_assert("cli_parse: --backend cuda must set no_metal = true", settings.no_metal);
    mu_assert("cli_parse: --backend cuda must NOT set no_cuda", !settings.no_cuda);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_backend_cpu()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "cpu"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend cpu must set no_cuda = true", settings.no_cuda);
    mu_assert("cli_parse: --backend cpu must set no_sycl = true", settings.no_sycl);
    mu_assert("cli_parse: --backend cpu must set no_vulkan = true", settings.no_vulkan);
    mu_assert("cli_parse: --backend cpu must set no_hip = true", settings.no_hip);
    mu_assert("cli_parse: --backend cpu must set no_metal = true", settings.no_metal);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_backend_sycl()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "sycl"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend sycl must set no_cuda = true", settings.no_cuda);
    mu_assert("cli_parse: --backend sycl must set no_vulkan = true", settings.no_vulkan);
    mu_assert("cli_parse: --backend sycl must set no_hip = true", settings.no_hip);
    mu_assert("cli_parse: --backend sycl must set no_metal = true", settings.no_metal);
    mu_assert("cli_parse: --backend sycl must default sycl_device to 0", settings.sycl_device == 0);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_backend_vulkan()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "vulkan"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend vulkan must set no_cuda = true", settings.no_cuda);
    mu_assert("cli_parse: --backend vulkan must set no_sycl = true", settings.no_sycl);
    mu_assert("cli_parse: --backend vulkan must set no_hip = true", settings.no_hip);
    mu_assert("cli_parse: --backend vulkan must set no_metal = true", settings.no_metal);
    mu_assert("cli_parse: --backend vulkan must default vulkan_device to 0",
              settings.vulkan_device == 0);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_backend_hip()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "hip"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend hip must set no_cuda = true", settings.no_cuda);
    mu_assert("cli_parse: --backend hip must set no_sycl = true", settings.no_sycl);
    mu_assert("cli_parse: --backend hip must set no_vulkan = true", settings.no_vulkan);
    mu_assert("cli_parse: --backend hip must set no_metal = true", settings.no_metal);
    mu_assert("cli_parse: --backend hip must NOT set no_hip", !settings.no_hip);
    mu_assert("cli_parse: --backend hip must default hip_device to 0", settings.hip_device == 0);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_backend_metal()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--backend", "metal"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --backend metal must set no_cuda = true", settings.no_cuda);
    mu_assert("cli_parse: --backend metal must set no_sycl = true", settings.no_sycl);
    mu_assert("cli_parse: --backend metal must set no_vulkan = true", settings.no_vulkan);
    mu_assert("cli_parse: --backend metal must set no_hip = true", settings.no_hip);
    mu_assert("cli_parse: --backend metal must NOT set no_metal", !settings.no_metal);
    mu_assert("cli_parse: --backend metal must default metal_device to 0",
              settings.metal_device == 0);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_hip_device_explicit()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--hip_device", "2"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --hip_device 2 must set hip_device = 2", settings.hip_device == 2);
    mu_assert("cli_parse: --hip_device must not engage no_hip", !settings.no_hip);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_metal_device_explicit()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--metal_device", "1"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --metal_device 1 must set metal_device = 1", settings.metal_device == 1);
    mu_assert("cli_parse: --metal_device must not engage no_metal", !settings.no_metal);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *test_no_hip_no_metal_flags()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--no_hip", "--no_metal"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --no_hip must set no_hip = true", settings.no_hip);
    mu_assert("cli_parse: --no_metal must set no_metal = true", settings.no_metal);
    mu_assert("cli_parse: --no_hip must leave hip_device at default -1", settings.hip_device == -1);
    mu_assert("cli_parse: --no_metal must leave metal_device at default -1",
              settings.metal_device == -1);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

/* Regression for audit finding F1 / ADR-0438: '-c' is declared in
 * short_opts[] but the switch previously had no 'case 'c'' arm, so
 * getopt_long consumed the option value and the switch fell into
 * default:, silently discarding the cpumask.  The fix adds a
 * 'case 'c':' fall-through before ARG_CPUMASK. */
static char *test_cpumask_short_opt()
{
    /* -c 0xff must set settings.cpumask = 255, same as --cpumask 0xff. */
    char *argv[9] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "-c", "0xff"};
    int argc = 7;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: -c 0xff must set cpumask = 255 (was silently dropped before ADR-0438)",
              settings.cpumask == 255);
    cli_free(&settings);
    cli_free_dicts(&settings);

    /* Decimal value: -c 3 */
    char *argv2[9] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "-c", "3"};
    int argc2 = 7;
    CLISettings settings2;
    optind = 1;
    cli_parse(argc2, argv2, &settings2);
    mu_assert("cli_parse: -c 3 must set cpumask = 3", settings2.cpumask == 3);
    cli_free(&settings2);
    cli_free_dicts(&settings2);

    return NULL;
}

/* Explicit `--gpumask=N --backend cuda` must preserve the user's gpumask,
 * NOT clobber it. Multi-GPU rigs need fine-grained disable bits. */
static char *test_backend_cuda_preserves_explicit_gpumask()
{
    char *argv[8] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--gpumask=2", "--backend", "cuda"};
    int argc = 8;
    CLISettings settings;
    optind = 1;
    cli_parse(argc, argv, &settings);
    mu_assert("cli_parse: --gpumask=2 --backend cuda must preserve gpumask = 2",
              settings.gpumask == 2);
    mu_assert("cli_parse: --gpumask=2 --backend cuda must keep use_gpumask = true",
              settings.use_gpumask);
    cli_free(&settings);
    cli_free_dicts(&settings);
    return NULL;
}

static char *run_aom_ctc_tests(void)
{
    mu_run_test(test_aom_ctc_v1_0);
    mu_run_test(test_aom_ctc_v2_0);
    mu_run_test(test_aom_ctc_v3_0);
    mu_run_test(test_aom_ctc_v4_0);
    mu_run_test(test_aom_ctc_v5_0);
    mu_run_test(test_aom_ctc_v6_0);
    mu_run_test(test_nflx_ctc_v1_0);
    return NULL;
}

static char *run_backend_tests(void)
{
    mu_run_test(test_backend_cpu);
    mu_run_test(test_backend_cuda_engages_cuda);
    mu_run_test(test_backend_cuda_preserves_explicit_gpumask);
    mu_run_test(test_backend_sycl);
    mu_run_test(test_backend_vulkan);
    mu_run_test(test_backend_hip);
    mu_run_test(test_backend_metal);
    mu_run_test(test_hip_device_explicit);
    mu_run_test(test_metal_device_explicit);
    mu_run_test(test_no_hip_no_metal_flags);
    mu_run_test(test_cpumask_short_opt);
    return NULL;
}

char *run_tests()
{
    char *result = run_aom_ctc_tests();
    if (result)
        return result;
    result = run_backend_tests();
    if (result)
        return result;
    return NULL;
}
