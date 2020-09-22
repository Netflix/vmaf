#include <assert.h>
#include <getopt.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cli_parse.h"

#include "libvmaf/feature.h"
#include "libvmaf/libvmaf.rc.h"
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
};

static const struct option long_opts[] = {
    { "reference",        1, NULL, 'r' },
    { "distorted",        1, NULL, 'd' },
    { "width",            1, NULL, 'w' },
    { "height",           1, NULL, 'h' },
    { "pixel_format",     1, NULL, 'p' },
    { "bitdepth",         1, NULL, 'b' },
    { "model",            1, NULL, 'm' },
    { "model_collection", 1, NULL, 'c' },
    { "output",           1, NULL, 'o' },
    { "xml",              0, NULL, ARG_OUTPUT_XML },
    { "json",             0, NULL, ARG_OUTPUT_JSON },
    { "csv",              0, NULL, ARG_OUTPUT_CSV },
    { "sub",              0, NULL, ARG_OUTPUT_SUB },
    { "threads",          1, NULL, ARG_THREADS },
    { "feature",          1, NULL, ARG_FEATURE },
    { "subsample",        1, NULL, ARG_SUBSAMPLE },
    { "cpumask",          1, NULL, ARG_CPUMASK },
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
            " --reference/-r $path:      path to reference .y4m or .yuv\n"
            " --distorted/-d $path:      path to distorted .y4m or .yuv\n"
            " --width/-w $unsigned:      width\n"
            " --height/-h $unsigned:     height\n"
            " --pixel_format/-p: $string pixel format (420/422/444)\n"
            " --bitdepth/-b $unsigned:   bitdepth (8/10/12)\n"
            " --model/-m $params:        model parameters, separated by the colon sign \":\"\n"
            "                            1. path to model file (required)\n"
            "                            2. name of model (required only if >1 models are used)\n"
            "                            3. other optional model parameters, e.g.\n"
            "                               path=foo.pkl:disable_clip\n"
            "                               path=foo.pkl:name=foo:enable_transform\n"
            " --model_collection/-c      model parameters. See `--model/-m` for details\n"
            " --output/-o $path:         path to output file\n"
            " --xml:                     write output file as XML (default)\n"
            " --json:                    write output file as JSON\n"
            " --csv:                     write output file as CSV\n"
            " --sub:                     write output file as subtitle\n"
            " --threads $unsigned:       number of threads to use\n"
            " --feature $string:         additional feature\n"
            " --cpumask: $bitmask        restrict permitted CPU instruction sets\n"
            " --subsample: $unsigned     compute scores only every N frames\n"
            " --quiet/-q:                disable FPS meter when run in a TTY\n"
            " --no_prediction/-n:        no prediction, extract features only\n"
            " --version/-v:              print version and exit\n"
           );
    exit(1);
}

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

static CLIModelConfig parse_model_config(const char *const optarg,
                                         const char *const app)
{
    char *optarg_copy = (char *)optarg;
    CLIModelConfig cli_cfg = {
        .cfg = { .flags = VMAF_MODEL_FLAGS_DEFAULT, },
        .path = NULL,
    };

    char *token;
    char delim[] = "=:";
    token = strtok(optarg_copy, delim);

    while (token != 0) {
        if(!strcmp(token, "path")) {
            cli_cfg.path = strtok(0, delim);
        } else if (!strcmp(token, "name")) {
            cli_cfg.cfg.name = strtok(0, delim);
        } else if (!strcmp(token, "disable_clip")) {
            cli_cfg.cfg.flags |= VMAF_MODEL_FLAG_DISABLE_CLIP;
        } else if (!strcmp(token, "enable_transform")) {
            cli_cfg.cfg.flags |= VMAF_MODEL_FLAG_ENABLE_TRANSFORM;
        } else if (!strcmp(token, "enable_ci")) {
            cli_cfg.cfg.flags |= VMAF_MODEL_FLAG_ENABLE_CONFIDENCE_INTERVAL;
        } else {
            usage(app, "Unknown parameter %s for model.\n", token);
        }
        token = strtok(0, delim);
    }

    if (!cli_cfg.path)
        usage(app, "For every model, path needs to be set.\n");

    return cli_cfg;
}

static CLIFeatureConfig parse_feature_config(const char *const optarg,
                                             const char *const app)
{
    char *optarg_copy = (char *)optarg;
    CLIFeatureConfig feature_cfg = {
        .name = strsep(&optarg_copy, "="),
        .opts_dict = NULL,
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
            if (settings->model_cnt == CLI_SETTINGS_STATIC_ARRAY_LEN) {
                usage(argv[0], "A maximum of %d models are supported\n",
                      CLI_SETTINGS_STATIC_ARRAY_LEN);
            }
            settings->model_config[settings->model_cnt++] =
                parse_model_config(optarg, argv[0]);
            break;
        case 'c':
            if (settings->model_collection_cnt == CLI_SETTINGS_STATIC_ARRAY_LEN) {
                usage(argv[0], "A maximum of %d model collections are supported\n",
                      CLI_SETTINGS_STATIC_ARRAY_LEN);
            }
            settings->model_collection_config[settings->model_collection_cnt++] =
                parse_model_config(optarg, argv[0]);
            break;
        case ARG_FEATURE:
            if (settings->feature_cnt == CLI_SETTINGS_STATIC_ARRAY_LEN) {
                usage(argv[0], "A maximum of %d features is supported\n",
                      CLI_SETTINGS_STATIC_ARRAY_LEN);
            }
            settings->feature_cfg[settings->feature_cnt++] =
                parse_feature_config(optarg, argv[0]);
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
    if (!settings->path_ref)
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
    if ((settings->model_cnt == 0 && settings->model_collection_cnt == 0) &&
        !settings->no_prediction)
    {
        usage(argv[0], "At least one model or model_collection "
                       "(-m/--model, -c/--model_collection) is required");
    }
}
