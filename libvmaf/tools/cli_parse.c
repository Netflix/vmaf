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

#include <assert.h>
#ifdef _WIN32
#include "compat/win32/getopt.h"
#else
#include <getopt.h>
#endif
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cli_parse.h"

#include "config.h"
#include "libvmaf/feature.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"

static const char short_opts[] = "r:d:w:h:p:b:m:c:o:nvq";

enum {
    ARG_OUTPUT_XML = 256,
    ARG_OUTPUT_JSON,
    ARG_OUTPUT_CSV,
    ARG_OUTPUT_SUB,
    ARG_THREADS,
    ARG_FEATURE,
    ARG_SUBSAMPLE,
    ARG_CPUMASK,
    ARG_GPUMASK,
    ARG_AOM_CTC,
    ARG_NFLX_CTC,
    ARG_FRAME_CNT,
    ARG_FRAME_SKIP_REF,
    ARG_FRAME_SKIP_DIST,
    ARG_NO_CUDA,
    ARG_NO_SYCL,
    ARG_SYCL_DEVICE,
    ARG_NO_VULKAN,
    ARG_VULKAN_DEVICE,
    ARG_NO_HIP,
    ARG_HIP_DEVICE,
    ARG_NO_METAL,
    ARG_METAL_DEVICE,
    ARG_BACKEND,
    ARG_PRECISION,
    ARG_TINY_MODEL,
    ARG_TINY_DEVICE,
    ARG_TINY_THREADS,
    ARG_TINY_FP16,
    ARG_TINY_MODEL_VERIFY,
    ARG_NO_REFERENCE,
    ARG_DNN_EP,
};

/* Default matches Netflix's pre-fork output exactly so the CPU golden
 * gate passes without explicit flags (CLAUDE.md §8). Round-trip lossless
 * formatting is opt-in via --precision=max. See ADR-0119 (supersedes
 * ADR-0006). */
#define VMAF_DEFAULT_PRECISION_FMT "%.6f"
#define VMAF_LOSSLESS_PRECISION_FMT "%.17g"

static char precision_fmt_buf[16];

static const char *resolve_precision_fmt(const char *optarg, const char *app, CLISettings *s)
{
    if (!strcmp(optarg, "max") || !strcmp(optarg, "full")) {
        s->precision_max = true;
        return VMAF_LOSSLESS_PRECISION_FMT;
    }
    if (!strcmp(optarg, "legacy")) {
        /* `legacy` is now the default; keep the alias accepted so existing
         * scripts that pass it explicitly do not break. */
        s->precision_legacy = true;
        return VMAF_DEFAULT_PRECISION_FMT;
    }
    char *end;
    long n = strtol(optarg, &end, 10);
    if (*end || end == optarg || n < 1 || n > 17) {
        (void)fprintf(stderr,
                      "%s: --precision must be an integer 1..17, "
                      "or one of: max, full, legacy (got: %s)\n",
                      app, optarg);
        exit(1);
    }
    s->precision_n = (int)n;
    (void)snprintf(precision_fmt_buf, sizeof precision_fmt_buf, "%%.%ldg", n);
    return precision_fmt_buf;
}

static const struct option long_opts[] = {
    {"reference", 1, NULL, 'r'},
    {"distorted", 1, NULL, 'd'},
    {"width", 1, NULL, 'w'},
    {"height", 1, NULL, 'h'},
    {"pixel_format", 1, NULL, 'p'},
    {"bitdepth", 1, NULL, 'b'},
    {"model", 1, NULL, 'm'},
    {"output", 1, NULL, 'o'},
    {"xml", 0, NULL, ARG_OUTPUT_XML},
    {"json", 0, NULL, ARG_OUTPUT_JSON},
    {"csv", 0, NULL, ARG_OUTPUT_CSV},
    {"sub", 0, NULL, ARG_OUTPUT_SUB},
    {"threads", 1, NULL, ARG_THREADS},
    {"feature", 1, NULL, ARG_FEATURE},
    {"subsample", 1, NULL, ARG_SUBSAMPLE},
    {"cpumask", 1, NULL, ARG_CPUMASK},
    {"gpumask", 1, NULL, ARG_GPUMASK},
    {"aom_ctc", 1, NULL, ARG_AOM_CTC},
    {"nflx_ctc", 1, NULL, ARG_NFLX_CTC},
    {"frame_cnt", 1, NULL, ARG_FRAME_CNT},
    {"frame_skip_ref", 1, NULL, ARG_FRAME_SKIP_REF},
    {"frame_skip_dist", 1, NULL, ARG_FRAME_SKIP_DIST},
    {"no_cuda", 0, NULL, ARG_NO_CUDA},
    {"no_sycl", 0, NULL, ARG_NO_SYCL},
    {"sycl_device", 1, NULL, ARG_SYCL_DEVICE},
    {"no_vulkan", 0, NULL, ARG_NO_VULKAN},
    {"vulkan_device", 1, NULL, ARG_VULKAN_DEVICE},
    {"no_hip", 0, NULL, ARG_NO_HIP},
    {"hip_device", 1, NULL, ARG_HIP_DEVICE},
    {"no_metal", 0, NULL, ARG_NO_METAL},
    {"metal_device", 1, NULL, ARG_METAL_DEVICE},
    {"backend", 1, NULL, ARG_BACKEND},
    {"precision", 1, NULL, ARG_PRECISION},
    {"tiny-model", 1, NULL, ARG_TINY_MODEL},
    {"tiny_model", 1, NULL, ARG_TINY_MODEL},
    {"tiny-device", 1, NULL, ARG_TINY_DEVICE},
    {"tiny_device", 1, NULL, ARG_TINY_DEVICE},
    {"tiny-threads", 1, NULL, ARG_TINY_THREADS},
    {"tiny_threads", 1, NULL, ARG_TINY_THREADS},
    {"tiny-fp16", 0, NULL, ARG_TINY_FP16},
    {"tiny_fp16", 0, NULL, ARG_TINY_FP16},
    {"tiny-model-verify", 0, NULL, ARG_TINY_MODEL_VERIFY},
    {"tiny_model_verify", 0, NULL, ARG_TINY_MODEL_VERIFY},
    {"no-reference", 0, NULL, ARG_NO_REFERENCE},
    {"no_reference", 0, NULL, ARG_NO_REFERENCE},
    /* --dnn-ep is the user-facing name for selecting the ONNX Runtime
     * execution provider. It is an alias for --tiny-device so both flags
     * write to the same CLISettings.tiny_device field. Accepting both names
     * lets users follow the ORT "execution provider" terminology directly
     * without knowing the fork's internal "tiny-device" naming. */
    {"dnn-ep", 1, NULL, ARG_DNN_EP},
    {"dnn_ep", 1, NULL, ARG_DNN_EP},
    {"no_prediction", 0, NULL, 'n'},
    {"version", 0, NULL, 'v'},
    {"quiet", 0, NULL, 'q'},
    {NULL, 0, NULL, 0},
};

_Noreturn static void usage(const char *const app, const char *const reason, ...);
static void usage(const char *const app, const char *const reason, ...)
{
    if (reason) {
        va_list args;
        va_start(args, reason);
        (void)vfprintf(stderr, reason, args);
        va_end(args);
        (void)fprintf(stderr, "\n\n");
    }
    (void)fprintf(stderr, "Usage: %s [options]\n\n", app);
    (void)fprintf(
        stderr,
        "Supported options:\n"
        " --reference/-r $path:        path to reference .y4m or .yuv\n"
        " --distorted/-d $path:        path to distorted .y4m or .yuv\n"
        " --width/-w $unsigned:        width\n"
        " --height/-h $unsigned:       height\n"
        " --pixel_format/-p: $string   pixel format (420/422/444)\n"
        " --bitdepth/-b $unsigned:     bitdepth (8/10/12/16)\n"
        " --model/-m $params:          model parameters, colon \":\" delimited\n"
        "                              `path=` path to model file\n"
        "                              `version=` built-in model version\n"
        "                              `name=` name used in log (optional)\n"
        " --output/-o $path:           output file\n"
        " --xml:                       write output file as XML (default)\n"
        " --json:                      write output file as JSON\n"
        " --csv:                       write output file as CSV\n"
        " --sub:                       write output file as subtitle\n"
        " --threads $unsigned:         number of threads to use\n"
        " --feature $string:           additional feature\n"
        " --cpumask: $bitmask          restrict permitted CPU instruction sets\n"
        " --gpumask: $bitmask          restrict permitted GPU operations\n"
        " --frame_cnt $unsigned:       maximum number of frames to process\n"
        " --frame_skip_ref $unsigned:  skip the first N frames in reference\n"
        " --frame_skip_dist $unsigned: skip the first N frames in distorted\n"
        " --subsample: $unsigned       compute scores only every N frames\n"
        " --no_cuda:                   disable CUDA backend\n"
        " --no_sycl:                    disable SYCL/oneAPI backend\n"
        " --sycl_device $unsigned:      select SYCL GPU by index (default: auto)\n"
        " --no_vulkan:                  disable Vulkan backend\n"
        " --vulkan_device $unsigned:    select Vulkan GPU by index (default: auto)\n"
        " --no_hip:                     disable HIP (AMD ROCm) backend\n"
        " --hip_device $unsigned:       select HIP GPU by index (default: auto)\n"
        " --no_metal:                   disable Metal (Apple Silicon) backend\n"
        " --metal_device $unsigned:     select Metal GPU by index (default: auto)\n"
        " --backend $name:              exclusive backend selector — auto|cpu|cuda|sycl|vulkan|hip|metal.\n"
        "                               When set to a specific backend, the others are\n"
        "                               disabled to avoid the dispatcher first-match-wins\n"
        "                               race that silently routes to CUDA when both Vulkan\n"
        "                               and CUDA are active.\n"
        " --precision $spec:            score output precision\n"
        "                                  N (1..17) -> printf \"%%.<N>g\"\n"
        "                                  max|full  -> \"%%.17g\" (round-trip lossless)\n"
        "                                  legacy    -> \"%%.6f\" (default; Netflix-compatible)\n"
        " --tiny-model $path:           load a tiny ONNX model alongside classic models\n"
        " --tiny-device $string:        auto|cpu|cuda|openvino|openvino-npu|\n"
        "                                  openvino-cpu|openvino-gpu|coreml|\n"
        "                                  coreml-ane|coreml-gpu|coreml-cpu|rocm\n"
        "                                  (default: auto)\n"
        " --dnn-ep $string:             alias for --tiny-device; selects the ONNX Runtime\n"
        "                                  execution provider by its ORT name\n"
        " --tiny-threads $unsigned:     CPU EP intra-op threads (0 = ORT default)\n"
        " --tiny-fp16:                  request fp16 IO where the EP supports it\n"
        " --tiny-model-verify:          require Sigstore-bundle verification (cosign verify-blob)\n"
        "                               of the loaded tiny model before use; refuses to load\n"
        "                               on missing bundle, missing cosign, or non-zero exit\n"
        " --no-reference:               no-reference mode; valid only with an NR tiny model\n"
        " --quiet/-q:                  disable FPS meter when run in a TTY\n"
        " --no_prediction/-n:          no prediction, extract features only\n"
        " --version/-v:                print version and exit\n");
    exit(1);
}

#define CHECKED_APPEND(arr, cnt, val, app, desc)                                                   \
    do {                                                                                           \
        if ((cnt) == CLI_SETTINGS_STATIC_ARRAY_LEN)                                                \
            usage((app), "A maximum of %d %s are supported\n", CLI_SETTINGS_STATIC_ARRAY_LEN,      \
                  (desc));                                                                         \
        (arr)[(cnt)++] = (val);                                                                    \
    } while (0)

#define CHECKED_REPLACE(arr, cnt, val, app, desc)                                                  \
    do {                                                                                           \
        CLIFeatureConfig _val = (val);                                                             \
        unsigned _i;                                                                               \
        for (_i = 0; _i < (cnt); _i++)                                                             \
            if (!strcmp((arr)[_i].name, _val.name)) {                                              \
                free((arr)[_i].buf);                                                               \
                vmaf_feature_dictionary_free(&(arr)[_i].opts_dict);                                \
                (arr)[_i] = _val;                                                                  \
                break;                                                                             \
            }                                                                                      \
        if (_i == (cnt))                                                                           \
            CHECKED_APPEND((arr), (cnt), _val, (app), (desc));                                     \
    } while (0)

static void error(const char *const app, const char *const optarg, const int option,
                  const char *const shouldbe)
{
    char optname[256];
    int n;

    for (n = 0; long_opts[n].name; n++) {
        if (long_opts[n].val == option)
            break;
    }
    assert(long_opts[n].name);
    if (long_opts[n].val < 256) {
        (void)sprintf(optname, "-%c/--%s", long_opts[n].val, long_opts[n].name);
    } else {
        (void)sprintf(optname, "--%s", long_opts[n].name);
    }

    usage(app, "Invalid argument \"%s\" for option %s; should be %s", optarg, optname, shouldbe);
}

static unsigned parse_unsigned(const char *const optarg, const int option, const char *const app)
{
    char *end;
    const unsigned res = (unsigned)strtoul(optarg, &end, 0);
    if (*end || end == optarg)
        error(app, optarg, option, "an integer");
    return res;
}

static unsigned parse_bitdepth(const char *const optarg, const int option, const char *const app)
{
    unsigned bitdepth = parse_unsigned(optarg, option, app);
    if (!((bitdepth == 8) || (bitdepth == 10) || (bitdepth == 12) || (bitdepth == 16)))
        error(app, optarg, option, "a valid bitdepth (8/10/12/16)");
    return bitdepth;
}

static enum VmafPixelFormat parse_pix_fmt(const char *const optarg, const int option,
                                          const char *const app)
{
    enum VmafPixelFormat pix_fmt = VMAF_PIX_FMT_UNKNOWN;

    if (!strcmp(optarg, "420"))
        pix_fmt = VMAF_PIX_FMT_YUV420P;
    if (!strcmp(optarg, "422"))
        pix_fmt = VMAF_PIX_FMT_YUV422P;
    if (!strcmp(optarg, "444"))
        pix_fmt = VMAF_PIX_FMT_YUV444P;

    if (!pix_fmt) {
        error(app, optarg, option,
              "a valid pixel format "
              "(420/422/444)");
    }

    return pix_fmt;
}

#ifndef HAVE_STRSEP
static char *strsep(char **sp, char *sep)
{
    char *p, *s;
    if (sp == NULL || *sp == NULL || **sp == '\0')
        return NULL;
    s = *sp;
    p = s + strcspn(s, sep);
    if (*p != '\0')
        *p++ = '\0';
    *sp = p;
    return s;
}
#endif

static CLIModelConfig parse_model_config(const char *const optarg, const char *const app)
{
    const size_t optarg_sz = strnlen(optarg, 1024);
    char *optarg_copy = malloc(optarg_sz + 1);
    if (!optarg_copy)
        usage(app, "error while parsing model option: %s", optarg);
    memset(optarg_copy, 0, optarg_sz + 1);
    strncpy(optarg_copy, optarg, optarg_sz);

    CLIModelConfig model_cfg = {
        .cfg =
            {
                .name = "vmaf",
                .flags = VMAF_MODEL_FLAGS_DEFAULT,
            },
        .path = NULL,
        .version = NULL,
        .buf = optarg_copy,
    };

    char *key_val;
    while ((key_val = strsep(&optarg_copy, ":")) != NULL) {
        char *key = strsep(&key_val, "=");
        char *val = strsep(&key_val, "=");
        if (!val) {
            if (!strcmp(key, "disable_clip")) {
                val = "true";
            } else if (!strcmp(key, "enable_transform")) {
                val = "true";
            } else {
                usage(app,
                      "Problem parsing model, "
                      "bad option string \"%s\".",
                      key);
            }
        }

        if (!strcmp(key, "path")) {
            model_cfg.path = val;
        } else if (!strcmp(key, "name")) {
            model_cfg.cfg.name = val;
        } else if (!strcmp(key, "version")) {
            model_cfg.version = val;
        } else if (!strcmp(key, "disable_clip")) {
            model_cfg.cfg.flags |= !strcmp(val, "true") ? VMAF_MODEL_FLAG_DISABLE_CLIP : 0;
        } else if (!strcmp(key, "enable_transform")) {
            model_cfg.cfg.flags |= !strcmp(val, "true") ? VMAF_MODEL_FLAG_ENABLE_TRANSFORM : 0;
        } else {
            if (model_cfg.overload_cnt == CLI_SETTINGS_STATIC_ARRAY_LEN) {
                usage(app,
                      "A maximum of %d feature overloads per model"
                      " are supported\n",
                      CLI_SETTINGS_STATIC_ARRAY_LEN);
            }
            char *name = strsep(&key, ".");
            model_cfg.feature_overload[model_cfg.overload_cnt].name = name;
            char *opt = strsep(&key, ".");
            int err = vmaf_feature_dictionary_set(
                &model_cfg.feature_overload[model_cfg.overload_cnt].opts_dict, opt, val);
            if (err)
                usage(app, "Problem parsing model: \"%s\"\n", name);

            model_cfg.overload_cnt++;
        }
    }

    return model_cfg;
}

/* CLI alias map: user-facing "integer_*" names to the internal extractor
 * registration names.  Extractors register without the "integer_" prefix (or
 * with a completely different name), so passing the prefix verbatim yields
 * "problem loading feature extractor".  Adding the map here — at the parse
 * layer — keeps the rewrite in one place and leaves the extractor registry
 * unchanged.  See the commit that introduced this table for the full list of
 * affected names. */
static const struct {
    const char *alias;
    const char *target;
} cli_feature_aliases[] = {
    {"integer_motion", "motion"}, {"integer_motion2", "motion_v2"},
    {"integer_ssim", "ssim"},     {"integer_ms_ssim", "float_ms_ssim"},
    {"integer_psnr", "psnr"},
};

static CLIFeatureConfig parse_feature_config(const char *const optarg, const char *const app)
{
    const size_t optarg_sz = strnlen(optarg, 1024);
    char *optarg_copy = malloc(optarg_sz + 1);
    if (!optarg_copy)
        usage(app, "error while parsing feature option: %s", optarg);
    memset(optarg_copy, 0, optarg_sz + 1);
    strncpy(optarg_copy, optarg, optarg_sz);
    void *buf = optarg_copy;

    CLIFeatureConfig feature_cfg = {
        .name = strsep(&optarg_copy, "="),
        .opts_dict = NULL,
        .buf = buf,
    };

    /* Rewrite user-facing "integer_*" aliases to the names the extractor
     * registry actually uses.  The rewrite only touches the name field; any
     * key=value options that follow the "=" separator are unaffected. */
    for (unsigned ai = 0; ai < sizeof(cli_feature_aliases) / sizeof(cli_feature_aliases[0]); ai++) {
        if (!strcmp(feature_cfg.name, cli_feature_aliases[ai].alias)) {
            feature_cfg.name = cli_feature_aliases[ai].target;
            break;
        }
    }

    char *key_val;
    while ((key_val = strsep(&optarg_copy, ":")) != NULL) {
        const char *key = strsep(&key_val, "=");
        const char *val = strsep(&key_val, "=");
        if (!val) {
            usage(app,
                  "Problem parsing feature \"%s\","
                  " bad option string \"%s\".\n",
                  feature_cfg.name, key);
        }
        int err = vmaf_feature_dictionary_set(&feature_cfg.opts_dict, key, val);
        if (err)
            usage(app, "Problem parsing feature \"%s\"\n", optarg);
    }

    return feature_cfg;
}

static void aom_ctc_v1_0(CLISettings *settings, const char *const app)
{
    CLIModelConfig cfg = {
        .version = "vmaf_v0.6.1",
        .cfg = {.name = "vmaf"},
    };
    CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg, app, "models");

    CLIModelConfig cfg_neg = {
        .version = "vmaf_v0.6.1neg",
        .cfg = {.name = "vmaf_neg"},
    };
    CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg_neg, app, "models");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
                   parse_feature_config("psnr=reduced_hbd_peak=true:"
                                        "enable_apsnr=true:min_sse=0.5",
                                        app),
                   app, "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt, parse_feature_config("ciede", app),
                   app, "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
                   parse_feature_config("float_ssim=enable_db=true:clip_db=true", app), app,
                   "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
                   parse_feature_config("float_ms_ssim=enable_db=true:clip_db=true", app), app,
                   "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
                   parse_feature_config("psnr_hvs", app), app, "features");
}

static void aom_ctc_v2_0(CLISettings *settings, const char *app)
{
    aom_ctc_v1_0(settings, app);
}

static void aom_ctc_v3_0(CLISettings *settings, const char *app)
{
    aom_ctc_v2_0(settings, app);
    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt, parse_feature_config("cambi", app),
                   app, "features");
}

static void aom_ctc_v4_0(CLISettings *settings, const char *app)
{
    aom_ctc_v3_0(settings, app);
}

static void aom_ctc_v5_0(CLISettings *settings, const char *app)
{
    aom_ctc_v4_0(settings, app);
}

static void aom_ctc_v6_0(CLISettings *settings, const char *app)
{
    aom_ctc_v5_0(settings, app);
    settings->common_bitdepth = true;
}

static void aom_ctc_v7_0(CLISettings *settings, const char *app)
{
    aom_ctc_v6_0(settings, app);
    CHECKED_REPLACE(settings->feature_cfg, settings->feature_cnt,
                    parse_feature_config("float_ssim=scale=1:enable_db=true:clip_db=true", app),
                    app, "features");
}

static void parse_aom_ctc(CLISettings *settings, const char *const optarg, const char *const app)
{
    if (!strcmp(optarg, "proposed"))
        usage(app, "`--aom_ctc proposed` is deprecated.");

    if (!strcmp(optarg, "v1.0")) {
        aom_ctc_v1_0(settings, app);
        return;
    }

    if (!strcmp(optarg, "v2.0")) {
        aom_ctc_v2_0(settings, app);
        return;
    }

    if (!strcmp(optarg, "v3.0")) {
        aom_ctc_v3_0(settings, app);
        return;
    }

    if (!strcmp(optarg, "v4.0")) {
        aom_ctc_v4_0(settings, app);
        return;
    }

    if (!strcmp(optarg, "v5.0")) {
        aom_ctc_v5_0(settings, app);
        return;
    }

    if (!strcmp(optarg, "v6.0")) {
        aom_ctc_v6_0(settings, app);
        return;
    }

    if (!strcmp(optarg, "v7.0")) {
        aom_ctc_v7_0(settings, app);
        return;
    }

    usage(app, "bad aom_ctc version \"%s\"", optarg);
}

static void nflx_ctc_v1_0(CLISettings *settings, const char *const app)
{
    CLIModelConfig cfg = {
        .version = "vmaf_4k_v0.6.1",
        .cfg = {.name = "vmaf"},
    };
    CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg, app, "models");

    CLIModelConfig cfg_neg = {
        .version = "vmaf_4k_v0.6.1neg",
        .cfg = {.name = "vmaf_neg"},
    };
    CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg_neg, app, "models");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
                   parse_feature_config("psnr=enable_chroma=true:enable_apsnr=true", app), app,
                   "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
                   parse_feature_config("float_ssim=enable_db=true:clip_db=true", app), app,
                   "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt, parse_feature_config("cambi", app),
                   app, "features");
}

static void parse_nflx_ctc(CLISettings *settings, const char *const optarg, const char *const app)
{
    if (!strcmp(optarg, "v1.0")) {
        nflx_ctc_v1_0(settings, app);
        return;
    }

    usage(app, "bad nflx_ctc version \"%s\"", optarg);
}

void cli_parse(const int argc, char *const *const argv, CLISettings *const settings)
{
    memset(settings, 0, sizeof(*settings));
    settings->sycl_device = -1;   // auto-select by default
    settings->vulkan_device = -1; // auto-select by default
    settings->hip_device = -1;    // auto-select by default
    settings->metal_device = -1;  // auto-select by default
    settings->precision_n = -1;
    settings->precision_fmt = VMAF_DEFAULT_PRECISION_FMT;
    settings->tiny_device = "auto";
    int o;

    while ((o = getopt_long(argc, argv, short_opts, long_opts, NULL)) >= 0) {
        switch (o) {
        case 'r':
            settings->path_ref = optarg;
            break;
        case 'd':
            settings->path_dist = optarg;
            break;
        case 'w':
            settings->width = parse_unsigned(optarg, 'w', argv[0]);
            settings->use_yuv = true;
            break;
        case 'h':
            settings->height = parse_unsigned(optarg, 'h', argv[0]);
            settings->use_yuv = true;
            break;
        case 'p':
            settings->pix_fmt = parse_pix_fmt(optarg, 'p', argv[0]);
            settings->use_yuv = true;
            break;
        case 'b':
            settings->bitdepth = parse_bitdepth(optarg, 'b', argv[0]);
            settings->use_yuv = true;
            break;
        case 'o':
            settings->output_path = optarg;
            break;
        case ARG_OUTPUT_XML:
            settings->output_fmt = VMAF_OUTPUT_FORMAT_XML;
            break;
        case ARG_OUTPUT_JSON:
            settings->output_fmt = VMAF_OUTPUT_FORMAT_JSON;
            break;
        case ARG_OUTPUT_CSV:
            settings->output_fmt = VMAF_OUTPUT_FORMAT_CSV;
            break;
        case ARG_OUTPUT_SUB:
            settings->output_fmt = VMAF_OUTPUT_FORMAT_SUB;
            break;
        case 'm':
            CHECKED_APPEND(settings->model_config, settings->model_cnt,
                           parse_model_config(optarg, argv[0]), argv[0], "models");
            break;
        case ARG_FEATURE:
            CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
                           parse_feature_config(optarg, argv[0]), argv[0], "features");
            break;
        /* The three handlers below pass the long-only enum value (not a
         * synthesised short-option char) to parse_unsigned() so that
         * error()'s walk over long_opts[] finds a matching entry. The
         * earlier 't' / 's' / 'c' shape tripped error()'s
         * assert(long_opts[n].name) for any non-numeric optarg
         * (e.g. `--threads abc`), turning a clean usage() error into a
         * SIGABRT — surfaced by the libFuzzer harness in PR #408
         * (ADR-0311). See ADR-0316.
         *
         * INVARIANT (ADR-0438): every short option declared in short_opts[]
         * must have a case arm in this switch.  The 'c' arm was absent until
         * the audit that produced ADR-0438 — getopt_long consumed -c <val>
         * from the command line but the switch fell into default: and silently
         * discarded the argument.  The fall-through below mirrors the
         * ARG_TINY_DEVICE / ARG_DNN_EP alias pattern already in this switch. */
        case ARG_THREADS:
            settings->thread_cnt = parse_unsigned(optarg, ARG_THREADS, argv[0]);
            break;
        case ARG_SUBSAMPLE:
            settings->subsample = parse_unsigned(optarg, ARG_SUBSAMPLE, argv[0]);
            break;
        case 'c':
        /* fall through — -c is the short form of --cpumask; both write
         * settings->cpumask via ARG_CPUMASK so error() reports the long
         * name on bad input. */
        case ARG_CPUMASK:
            settings->cpumask = parse_unsigned(optarg, ARG_CPUMASK, argv[0]);
            break;
        case ARG_GPUMASK:
            settings->gpumask = parse_unsigned(optarg, ARG_GPUMASK, argv[0]);
            settings->use_gpumask = true;
            break;
        case ARG_AOM_CTC:
            parse_aom_ctc(settings, optarg, argv[0]);
            break;
        case ARG_NFLX_CTC:
            parse_nflx_ctc(settings, optarg, argv[0]);
            break;
        case ARG_FRAME_CNT:
            settings->frame_cnt = parse_unsigned(optarg, ARG_FRAME_CNT, argv[0]);
            break;
        case ARG_FRAME_SKIP_REF:
            settings->frame_skip_ref = parse_unsigned(optarg, ARG_FRAME_SKIP_REF, argv[0]);
            break;
        case ARG_FRAME_SKIP_DIST:
            settings->frame_skip_dist = parse_unsigned(optarg, ARG_FRAME_SKIP_DIST, argv[0]);
            break;
        case ARG_NO_CUDA:
            settings->no_cuda = true;
            break;
        case ARG_NO_SYCL:
            settings->no_sycl = true;
            break;
        case ARG_SYCL_DEVICE:
            settings->sycl_device = (int)parse_unsigned(optarg, ARG_SYCL_DEVICE, argv[0]);
            break;
        case ARG_NO_VULKAN:
            settings->no_vulkan = true;
            break;
        case ARG_VULKAN_DEVICE:
            settings->vulkan_device = (int)parse_unsigned(optarg, ARG_VULKAN_DEVICE, argv[0]);
            break;
        case ARG_NO_HIP:
            settings->no_hip = true;
            break;
        case ARG_HIP_DEVICE:
            settings->hip_device = (int)parse_unsigned(optarg, ARG_HIP_DEVICE, argv[0]);
            break;
        case ARG_NO_METAL:
            settings->no_metal = true;
            break;
        case ARG_METAL_DEVICE:
            settings->metal_device = (int)parse_unsigned(optarg, ARG_METAL_DEVICE, argv[0]);
            break;
        case ARG_BACKEND:
            settings->backend = optarg;
            break;
        case ARG_PRECISION:
            settings->precision_fmt = resolve_precision_fmt(optarg, argv[0], settings);
            break;
        case ARG_TINY_MODEL:
            settings->tiny_model_path = optarg;
            break;
        case ARG_TINY_DEVICE:
        /* fall through — --dnn-ep is an alias; both write tiny_device */
        case ARG_DNN_EP:
            if (strcmp(optarg, "auto") && strcmp(optarg, "cpu") && strcmp(optarg, "cuda") &&
                strcmp(optarg, "openvino") && strcmp(optarg, "openvino-npu") &&
                strcmp(optarg, "openvino-cpu") && strcmp(optarg, "openvino-gpu") &&
                strcmp(optarg, "coreml") && strcmp(optarg, "coreml-ane") &&
                strcmp(optarg, "coreml-gpu") && strcmp(optarg, "coreml-cpu") &&
                strcmp(optarg, "rocm")) {
                error(argv[0], optarg, o == ARG_DNN_EP ? ARG_DNN_EP : ARG_TINY_DEVICE,
                      "one of auto|cpu|cuda|openvino|openvino-npu|openvino-cpu|"
                      "openvino-gpu|coreml|coreml-ane|coreml-gpu|coreml-cpu|rocm");
            }
            settings->tiny_device = optarg;
            break;
        case ARG_TINY_THREADS:
            settings->tiny_threads = (int)parse_unsigned(optarg, ARG_TINY_THREADS, argv[0]);
            break;
        case ARG_TINY_FP16:
            settings->tiny_fp16 = true;
            break;
        case ARG_TINY_MODEL_VERIFY:
            settings->tiny_model_verify = true;
            break;
        case ARG_NO_REFERENCE:
            settings->no_reference = true;
            break;
        case 'n':
            settings->no_prediction = true;
            break;
        case 'q':
            settings->quiet = true;
            break;
        case 'v':
            (void)fprintf(stderr, "%s\n", vmaf_version());
            exit(0);
        default:
            break;
        }
    }

    if (!settings->output_fmt)
        settings->output_fmt = VMAF_OUTPUT_FORMAT_XML;
    /* --backend exclusive selector. Apply BEFORE the rest of the
     * post-parse validation so the per-backend flags are consistent
     * downstream. Must run before any code path that consumes
     * settings->no_cuda / no_sycl / no_vulkan. */
    if (settings->backend) {
        if (!strcmp(settings->backend, "auto")) {
            /* Default — leave per-backend flags as-is. */
        } else if (!strcmp(settings->backend, "cpu")) {
            settings->no_cuda = true;
            settings->no_sycl = true;
            settings->no_vulkan = true;
            settings->no_hip = true;
            settings->no_metal = true;
        } else if (!strcmp(settings->backend, "cuda")) {
            settings->no_sycl = true;
            settings->no_vulkan = true;
            settings->no_hip = true;
            settings->no_metal = true;
            if (!settings->use_gpumask) {
                /* `gpumask` is a CUDA-*disable* bitmask per the public
                 * VmafConfiguration::gpumask contract — `compute_fex_flags`
                 * routes the CUDA dispatch slot only when `gpumask == 0`.
                 * Setting `use_gpumask = true` triggers `vmaf_cuda_state_init`
                 * in the CLI; leaving `gpumask = 0` lets the runtime then
                 * actually pick the CUDA extractors. Earlier revisions set
                 * `gpumask = 1` here intending it as a device-pin, which
                 * silently disabled CUDA and routed everything through the
                 * CPU path — see test_backend_cuda_engages_cuda below. */
                settings->gpumask = 0;
                settings->use_gpumask = true;
            }
        } else if (!strcmp(settings->backend, "sycl")) {
            settings->no_cuda = true;
            settings->no_vulkan = true;
            settings->no_hip = true;
            settings->no_metal = true;
            if (settings->sycl_device < 0)
                settings->sycl_device = 0;
        } else if (!strcmp(settings->backend, "vulkan")) {
            settings->no_cuda = true;
            settings->no_sycl = true;
            settings->no_hip = true;
            settings->no_metal = true;
            if (settings->vulkan_device < 0)
                settings->vulkan_device = 0;
        } else if (!strcmp(settings->backend, "hip")) {
            settings->no_cuda = true;
            settings->no_sycl = true;
            settings->no_vulkan = true;
            settings->no_metal = true;
            if (settings->hip_device < 0)
                settings->hip_device = 0;
        } else if (!strcmp(settings->backend, "metal")) {
            settings->no_cuda = true;
            settings->no_sycl = true;
            settings->no_vulkan = true;
            settings->no_hip = true;
            if (settings->metal_device < 0)
                settings->metal_device = 0;
        } else {
            usage(argv[0],
                  "Unknown --backend value '%s' "
                  "(expected: auto|cpu|cuda|sycl|vulkan|hip|metal)",
                  settings->backend);
        }
    }

    if (!settings->path_ref)
        usage(argv[0], "Reference .y4m or .yuv (-r/--reference) is required");
    if (!settings->path_dist)
        usage(argv[0], "Distorted .y4m or .yuv (-d/--distorted) is required");
    if (settings->use_yuv &&
        !(settings->width && settings->height && settings->pix_fmt && settings->bitdepth)) {
        usage(argv[0], "The following options are required for .yuv input:\n"
                       "  --width/-w\n"
                       "  --height/-h\n"
                       "  --pixel_format/-p\n"
                       "  --bitdepth/-b\n");
    }

    if (settings->model_cnt == 0 && !settings->no_prediction) {
#if VMAF_BUILT_IN_MODELS
        CLIModelConfig cfg = {
            .version = "vmaf_v0.6.1",
        };
        CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg, argv[0], "models");
#else
        usage(argv[0], "At least one model (-m/--model) is required "
                       "unless no prediction (-n/--no_prediction) is set");
#endif
    }

    for (unsigned i = 0; i < settings->model_cnt; i++) {
        for (unsigned j = 0; j < settings->model_cnt; j++) {
            if (i == j)
                continue;
            if (!strcmp(settings->model_config[i].cfg.name, settings->model_config[j].cfg.name)) {
                usage(argv[0], "Each model should be uniquely named. "
                               "Set using `--model` via the `name=...` param.");
            }
        }
    }
}

void cli_free(CLISettings *settings)
{
    for (unsigned i = 0; i < settings->model_cnt; i++)
        free(settings->model_config[i].buf);
    for (unsigned i = 0; i < settings->feature_cnt; i++)
        free(settings->feature_cfg[i].buf);
}
