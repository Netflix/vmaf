#include <assert.h>
#ifdef HAVE_GETOPT_H
#include <getopt.h>
#else
#error "Meson target is missing getopt_dependency"
#endif
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
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
    ARG_COLOR_RANGE_REF,
    ARG_COLOR_RANGE_DIST,
    ARG_COLOR_PRIMARIES_REF,
    ARG_COLOR_PRIMARIES_DIST,
    ARG_COLOR_TRC_REF,
    ARG_COLOR_TRC_DIST,
    ARG_COLOR_MATRIX_REF,
    ARG_COLOR_MATRIX_DIST,
};

static const struct option long_opts[] = {
    { "reference",        1, NULL, 'r' },
    { "distorted",        1, NULL, 'd' },
    { "width",            1, NULL, 'w' },
    { "height",           1, NULL, 'h' },
    { "pixel_format",     1, NULL, 'p' },
    { "bitdepth",         1, NULL, 'b' },
    { "model",            1, NULL, 'm' },
    { "output",           1, NULL, 'o' },
    { "xml",              0, NULL, ARG_OUTPUT_XML },
    { "json",             0, NULL, ARG_OUTPUT_JSON },
    { "csv",              0, NULL, ARG_OUTPUT_CSV },
    { "sub",              0, NULL, ARG_OUTPUT_SUB },
    { "threads",          1, NULL, ARG_THREADS },
    { "feature",          1, NULL, ARG_FEATURE },
    { "subsample",        1, NULL, ARG_SUBSAMPLE },
    { "cpumask",          1, NULL, ARG_CPUMASK },
    { "gpumask",          1, NULL, ARG_GPUMASK },
    { "aom_ctc",          1, NULL, ARG_AOM_CTC },
    { "nflx_ctc",         1, NULL, ARG_NFLX_CTC },
    { "frame_cnt",        1, NULL, ARG_FRAME_CNT },
    { "frame_skip_ref",   1, NULL, ARG_FRAME_SKIP_REF },
    { "frame_skip_dist",  1, NULL, ARG_FRAME_SKIP_DIST },
    { "color_range_ref",  1, NULL, ARG_COLOR_RANGE_REF },
    { "color_range_dist", 1, NULL, ARG_COLOR_RANGE_DIST },
    { "color_primaries_ref",  1, NULL, ARG_COLOR_PRIMARIES_REF },
    { "color_primaries_dist", 1, NULL, ARG_COLOR_PRIMARIES_DIST },
    { "color_trc_ref",    1, NULL, ARG_COLOR_TRC_REF },
    { "color_trc_dist",   1, NULL, ARG_COLOR_TRC_DIST },
    { "color_matrix_ref", 1, NULL, ARG_COLOR_MATRIX_REF },
    { "color_matrix_dist", 1, NULL, ARG_COLOR_MATRIX_DIST },
    { "no_prediction",    0, NULL, 'n' },
    { "version",          0, NULL, 'v' },
    { "quiet",            0, NULL, 'q' },
    { NULL,               0, NULL, 0 },
};

static void usage(const char *const app, const char *const reason, ...) {
    if (reason) {
        va_list args;
        va_start(args, reason);
        vfprintf(stderr, reason, args);
        va_end(args);
        fprintf(stderr, "\n\n");
    }
    fprintf(stderr, "Usage: %s [options]\n\n", app);
    fprintf(stderr, "Supported options:\n"
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
            " --color_range_ref/_dist $string:\n"
            "                              color range for the reference/distorted input\n"
            " --color_primaries_ref/_dist $string:\n"
            "                              color primaries for the reference/distorted input\n"
            " --color_trc_ref/_dist $string:\n"
            "                              transfer characteristic for the reference/distorted input\n"
            " --color_matrix_ref/_dist $string:\n"
            "                              matrix coefficients for the reference/distorted input\n"
            " --quiet/-q:                  disable FPS meter when run in a TTY\n"
            " --no_prediction/-n:          no prediction, extract features only\n"
            " --version/-v:                print version and exit\n"
           );
    exit(1);
}

#define CHECKED_APPEND(arr, cnt, val, app, desc) do {                   \
    if ((cnt) == CLI_SETTINGS_STATIC_ARRAY_LEN)                         \
        usage((app), "A maximum of %d %s are supported\n",              \
              CLI_SETTINGS_STATIC_ARRAY_LEN, (desc));                   \
    (arr)[(cnt)++] = (val);                                             \
} while (0)

#define CHECKED_REPLACE(arr, cnt, val, app, desc) do {                  \
    CLIFeatureConfig _val = (val);                                      \
    unsigned _i;                                                        \
    for (_i = 0; _i < (cnt); _i++)                                      \
        if (!strcmp((arr)[_i].name, _val.name)) {                       \
            free((arr)[_i].buf);                                        \
            vmaf_feature_dictionary_free(&(arr)[_i].opts_dict);         \
            (arr)[_i] = _val;                                           \
            break;                                                      \
        }                                                               \
    if (_i == (cnt))                                                    \
        CHECKED_APPEND((arr), (cnt), _val, (app), (desc));              \
} while (0)

static void error(const char *const app, const char *const optarg,
                  const int option, const char *const shouldbe)
{
    char optname[256];
    int n;

    for (n = 0; long_opts[n].name; n++)
        if (long_opts[n].val == option)
            break;
    assert(long_opts[n].name);
    if (long_opts[n].val < 256) {
        sprintf(optname, "-%c/--%s", long_opts[n].val, long_opts[n].name);
    } else {
        sprintf(optname, "--%s", long_opts[n].name);
    }

    usage(app, "Invalid argument \"%s\" for option %s; should be %s",
          optarg, optname, shouldbe);
}

static unsigned parse_unsigned(const char *const optarg, const int option,
                               const char *const app)
{
    char *end;
    const unsigned res = (unsigned) strtoul(optarg, &end, 0);
    if (*end || end == optarg) error(app, optarg, option, "an integer");
    return res;
}

static unsigned parse_bitdepth(const char *const optarg, const int option,
                               const char *const app)
{
    unsigned bitdepth = parse_unsigned(optarg, option, app);
    if (!((bitdepth == 8) || (bitdepth == 10) || (bitdepth == 12) || (bitdepth == 16)))
        error(app, optarg, option, "a valid bitdepth (8/10/12/16)");
    return bitdepth;
}

static enum VmafPixelFormat parse_pix_fmt(const char *const optarg,
                                          const int option,
                                          const char *const app)
{
    enum VmafPixelFormat pix_fmt = VMAF_PIX_FMT_UNKNOWN;

    if (!strcmp(optarg, "420"))
        pix_fmt = VMAF_PIX_FMT_YUV420P;
    if (!strcmp(optarg, "422"))
        pix_fmt = VMAF_PIX_FMT_YUV422P;
    if (!strcmp(optarg, "444"))
        pix_fmt = VMAF_PIX_FMT_YUV444P;

    if (!pix_fmt) error(app, optarg, option, "a valid pixel format "
                                             "(420/422/444)");

    return pix_fmt;
}

static enum VmafColorRange parse_color_range(const char *const optarg,
                                              const int option,
                                              const char *const app)
{
    if (!strcmp(optarg, "unknown")) return VMAF_COLOR_RANGE_UNKNOWN;
    if (!strcmp(optarg, "limited")) return VMAF_COLOR_RANGE_LIMITED;
    if (!strcmp(optarg, "full")) return VMAF_COLOR_RANGE_FULL;

    error(app, optarg, option, "a valid color range "
                               "(unknown/limited/full)");
    return VMAF_COLOR_RANGE_UNKNOWN;
}

typedef struct { const char *name; unsigned val; } NamedValue;

static int lookup_named_value(const NamedValue *table, size_t table_len,
                              const char *const optarg, unsigned *val)
{
    for (size_t i = 0; i < table_len; i++) {
        if (!strcmp(optarg, table[i].name)) {
            *val = table[i].val;
            return 0;
        }
    }
    return -1;
}

static const NamedValue color_primaries_names[] = {
    { "unknown",  VMAF_COLOR_PRIMARIES_UNKNOWN },
    { "bt709",    VMAF_COLOR_PRIMARIES_BT709 },
    { "bt2020",   VMAF_COLOR_PRIMARIES_BT2020 },
};

static const NamedValue color_trc_names[] = {
    { "unknown",       VMAF_COLOR_TRC_UNKNOWN },
    { "bt709",         VMAF_COLOR_TRC_BT709 },
    { "smpte2084",     VMAF_COLOR_TRC_SMPTE2084 },
    { "pq",            VMAF_COLOR_TRC_SMPTE2084 },
};

static const NamedValue color_matrix_names[] = {
    { "unknown",              VMAF_COLOR_MATRIX_UNKNOWN },
    { "bt709",                VMAF_COLOR_MATRIX_BT709 },
    { "bt2020nc",             VMAF_COLOR_MATRIX_BT2020_NCL },
    { "ictcp",                VMAF_COLOR_MATRIX_ICTCP },
};

static enum VmafColorPrimaries parse_color_primaries(
    const char *const optarg, const int option, const char *const app)
{
    unsigned val;
    if (lookup_named_value(color_primaries_names,
                           sizeof(color_primaries_names) /
                               sizeof(*color_primaries_names),
                           optarg, &val))
    {
        error(app, optarg, option,
              "a valid color primaries name, e.g. unknown/bt709/bt2020");
        return VMAF_COLOR_PRIMARIES_UNKNOWN;
    }
    return (enum VmafColorPrimaries) val;
}

static enum VmafColorTransferCharacteristic parse_color_trc(
    const char *const optarg, const int option, const char *const app)
{
    unsigned val;
    if (lookup_named_value(color_trc_names,
                           sizeof(color_trc_names) / sizeof(*color_trc_names),
                           optarg, &val))
    {
        error(app, optarg, option,
              "a valid transfer characteristic name, e.g. unknown/bt709/pq");
        return VMAF_COLOR_TRC_UNKNOWN;
    }
    return (enum VmafColorTransferCharacteristic) val;
}

static enum VmafColorMatrixCoefficients parse_color_matrix(
    const char *const optarg, const int option, const char *const app)
{
    unsigned val;
    if (lookup_named_value(color_matrix_names,
                           sizeof(color_matrix_names) /
                               sizeof(*color_matrix_names),
                           optarg, &val))
    {
        error(app, optarg, option, "a valid matrix coefficients name, e.g. "
                                   "bt709/bt2020nc/ictcp");
        return VMAF_COLOR_MATRIX_UNKNOWN;
    }
    return (enum VmafColorMatrixCoefficients) val;
}

/* --color_X_ref and --color_X_dist describe one input each. */
#define COLOR_ARG_CASES(attr, ARG, parse)                                   \
    case ARG##_REF:                                                         \
        settings->color_ref.attr = parse(optarg, ARG##_REF, argv[0]);       \
        break;                                                              \
    case ARG##_DIST:                                                        \
        settings->color_dist.attr = parse(optarg, ARG##_DIST, argv[0]);     \
        break;

static bool color_any_set(const VmafColor *color)
{
    return color->range != VMAF_COLOR_RANGE_UNKNOWN ||
           color->primaries != VMAF_COLOR_PRIMARIES_UNKNOWN ||
           color->trc != VMAF_COLOR_TRC_UNKNOWN ||
           color->matrix != VMAF_COLOR_MATRIX_UNKNOWN;
}

static bool color_all_set(const VmafColor *color)
{
    return color->range != VMAF_COLOR_RANGE_UNKNOWN &&
           color->primaries != VMAF_COLOR_PRIMARIES_UNKNOWN &&
           color->trc != VMAF_COLOR_TRC_UNKNOWN &&
           color->matrix != VMAF_COLOR_MATRIX_UNKNOWN;
}

#ifndef HAVE_STRSEP
static char *strsep(char **sp, char *sep)
{
    char *p, *s;
    if (sp == NULL || *sp == NULL || **sp == '\0') return NULL;
    s = *sp;
    p = s + strcspn(s, sep);
    if (*p != '\0') *p++ = '\0';
    *sp = p;
    return s;
}
#endif

// Like strsep(s, ":"), but a colon that follows a drive letter ("path=C:\\dir",
// "path=C:/dir") belongs to the value instead of ending the option.
static char *next_option(char **s)
{
    char *start = *s;
    if (!start) return NULL;
    for (char *c = start; *c; c++) {
        if (*c != ':') continue;
        const int drive = c - start >= 2 && c[-2] == '=' &&
                          isalpha((unsigned char)c[-1]) &&
                          (c[1] == '\\' || c[1] == '/');
        if (drive) continue;
        *c = '\0';
        *s = c + 1;
        return start;
    }
    *s = NULL;
    return start;
}

static CLIModelConfig parse_model_config(const char *const optarg,
                                         const char *const app)
{
    const size_t optarg_sz = strnlen(optarg, 1024);
    char *optarg_copy = malloc(optarg_sz + 1);
    if (!optarg_copy)
        usage(app, "error while parsing model option: %s", optarg);
    memset(optarg_copy, 0, optarg_sz + 1);
    strncpy(optarg_copy, optarg, optarg_sz);

    CLIModelConfig model_cfg = {
        .cfg = {
            .name = "vmaf",
            .flags = VMAF_MODEL_FLAGS_DEFAULT,
        },
        .path = NULL,
        .version = NULL,
        .buf = optarg_copy,
    };

    char *key_val;
    while ((key_val = next_option(&optarg_copy)) != NULL) {
        char *key = strsep(&key_val, "=");
        char *val = strsep(&key_val, "=");
        if (!val) {
            if (!strcmp(key, "disable_clip")) {
                val = "true";
            } else if (!strcmp(key, "enable_transform")) {
                val = "true";
            } else {
                usage(app, "Problem parsing model, "
                           "bad option string \"%s\".", key);
            }
        }

        if (!strcmp(key, "path")) {
            model_cfg.path = val;
        } else if (!strcmp(key, "name")) {
            model_cfg.cfg.name = val;
        } else if (!strcmp(key, "version")) {
            model_cfg.version = val;
        } else if (!strcmp(key, "disable_clip")) {
            model_cfg.cfg.flags |=
                !strcmp(val, "true") ? VMAF_MODEL_FLAG_DISABLE_CLIP : 0;
        } else if (!strcmp(key, "enable_transform")) {
            model_cfg.cfg.flags |=
                !strcmp(val, "true") ? VMAF_MODEL_FLAG_ENABLE_TRANSFORM : 0;
        } else {
            if (model_cfg.overload_cnt == CLI_SETTINGS_STATIC_ARRAY_LEN) {
                usage(app, "A maximum of %d feature overloads per model"
                      " are supported\n", CLI_SETTINGS_STATIC_ARRAY_LEN);
            }
            char *name = strsep(&key, ".");
            model_cfg.feature_overload[model_cfg.overload_cnt].name = name;
            char *opt = strsep(&key, ".");
            int err =
                vmaf_feature_dictionary_set(
                    &model_cfg.feature_overload[model_cfg.overload_cnt].opts_dict,
                    opt, val);
            if (err) usage(app, "Problem parsing model: \"%s\"\n", name);

            model_cfg.overload_cnt++;
        }
    }

    return model_cfg;
}

static CLIFeatureConfig parse_feature_config(const char *const optarg,
                                             const char *const app)
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

    char *key_val;
    while ((key_val = strsep(&optarg_copy, ":")) != NULL) {
        const char *key = strsep(&key_val, "=");
        const char *val = strsep(&key_val, "=");
        if (!val) {
            usage(app, "Problem parsing feature \"%s\","
                       " bad option string \"%s\".\n", feature_cfg.name, key);
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
        .cfg = { .name = "vmaf" },
    };
    CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg,
                   app, "models");

    CLIModelConfig cfg_neg = {
        .version = "vmaf_v0.6.1neg",
        .cfg = { .name = "vmaf_neg" },
    };
    CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg_neg,
                   app, "models");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
        parse_feature_config("psnr=reduced_hbd_peak=true:"
                             "enable_apsnr=true:min_sse=0.5", app),
        app, "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
        parse_feature_config("ciede", app), app, "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
        parse_feature_config("float_ssim=enable_db=true:clip_db=true", app),
        app, "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
        parse_feature_config("float_ms_ssim=enable_db=true:clip_db=true", app),
        app, "features");

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
    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
        parse_feature_config("cambi", app), app, "features");
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
        parse_feature_config("float_ssim=scale=1:enable_db=true:clip_db=true",
                             app),
        app, "features");
}

static void parse_aom_ctc(CLISettings *settings, const char *const optarg,
                          const char *const app)
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
        .cfg = { .name = "vmaf" },
    };
    CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg,
                   app, "models");

    CLIModelConfig cfg_neg = {
        .version = "vmaf_4k_v0.6.1neg",
        .cfg = { .name = "vmaf_neg" },
    };
    CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg_neg,
                   app, "models");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
        parse_feature_config("psnr=enable_chroma=true:enable_apsnr=true", app),
        app, "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
        parse_feature_config("float_ssim=enable_db=true:clip_db=true", app),
        app, "features");

    CHECKED_APPEND(settings->feature_cfg, settings->feature_cnt,
        parse_feature_config("cambi", app), app, "features");
}

static void parse_nflx_ctc(CLISettings *settings, const char *const optarg,
                         const char *const app)
{
    if (!strcmp(optarg, "v1.0")) {
        nflx_ctc_v1_0(settings, app);
        return;
    }

    usage(app, "bad nflx_ctc version \"%s\"", optarg);
}

void cli_parse(const int argc, char *const *const argv,
               CLISettings *const settings)
{
    memset(settings, 0, sizeof(*settings));
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
        case ARG_THREADS:
            settings->thread_cnt = parse_unsigned(optarg, 't', argv[0]);
            break;
        case ARG_SUBSAMPLE:
            settings->subsample = parse_unsigned(optarg, 's', argv[0]);
            break;
        case ARG_CPUMASK:
            settings->cpumask = parse_unsigned(optarg, 'c', argv[0]);
            break;
        case ARG_GPUMASK:
            settings->gpumask = parse_unsigned(optarg, ARG_GPUMASK, argv[0]);
            break;
        case ARG_AOM_CTC:
            parse_aom_ctc(settings, optarg, argv[0]);
            break;
        case ARG_NFLX_CTC:
            parse_nflx_ctc(settings, optarg, argv[0]);
            break;
        case ARG_FRAME_CNT:
            settings->frame_cnt =
                parse_unsigned(optarg, ARG_FRAME_CNT, argv[0]);
            break;
        case ARG_FRAME_SKIP_REF:
            settings->frame_skip_ref = parse_unsigned(optarg, ARG_FRAME_SKIP_REF, argv[0]);
            break;
        case ARG_FRAME_SKIP_DIST:
            settings->frame_skip_dist = parse_unsigned(optarg, ARG_FRAME_SKIP_DIST, argv[0]);
            break;
        COLOR_ARG_CASES(range, ARG_COLOR_RANGE, parse_color_range)
        COLOR_ARG_CASES(primaries, ARG_COLOR_PRIMARIES, parse_color_primaries)
        COLOR_ARG_CASES(trc, ARG_COLOR_TRC, parse_color_trc)
        COLOR_ARG_CASES(matrix, ARG_COLOR_MATRIX, parse_color_matrix)
        case 'n':
            settings->no_prediction = true;
            break;
        case 'q':
            settings->quiet = true;
            break;
        case 'v':
            fprintf(stderr, "%s\n", vmaf_version());
            exit(0);
        default:
            break;
        }
    }

    if (!settings->output_fmt)
        settings->output_fmt = VMAF_OUTPUT_FORMAT_XML;
    if (!settings->path_ref)
        usage(argv[0], "Reference .y4m or .yuv (-r/--reference) is required");
    if (!settings->path_dist)
        usage(argv[0], "Distorted .y4m or .yuv (-d/--distorted) is required");
    if (settings->use_yuv && !(settings->width && settings->height &&
        settings->pix_fmt && settings->bitdepth))
    {
        usage(argv[0], "The following options are required for .yuv input:\n"
                       "  --width/-w\n"
                       "  --height/-h\n"
                       "  --pixel_format/-p\n"
                       "  --bitdepth/-b\n");
    }

    const bool ref_color_set = color_all_set(&settings->color_ref);
    const bool dist_color_set = color_all_set(&settings->color_dist);
    if (color_any_set(&settings->color_ref) && !ref_color_set) {
        usage(argv[0], "The reference colorimetry must either be fully "
                       "specified or left unset: provide all of "
                       "--color_range_ref, --color_primaries_ref, "
                       "--color_trc_ref and --color_matrix_ref\n");
    }
    if (color_any_set(&settings->color_dist) && !dist_color_set) {
        usage(argv[0], "The distorted colorimetry must either be fully "
                       "specified or left unset: provide all of "
                       "--color_range_dist, --color_primaries_dist, "
                       "--color_trc_dist and --color_matrix_dist\n");
    }

    if (settings->model_cnt == 0 && !settings->no_prediction) {
#if VMAF_BUILT_IN_MODELS
        CLIModelConfig cfg = {
            .version = "vmaf_v0.6.1",
        };
        CHECKED_APPEND(settings->model_config, settings->model_cnt, cfg,
                       argv[0], "models");
#else
        usage(argv[0], "At least one model (-m/--model) is required "
                       "unless no prediction (-n/--no_prediction) is set");
#endif
    }

    for (unsigned i = 0; i < settings->model_cnt; i++) {
        for (unsigned j = 0; j < settings->model_cnt; j++) {
            if (i == j) continue;
            if (!strcmp(settings->model_config[i].cfg.name,
                        settings->model_config[j].cfg.name))
            {
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
