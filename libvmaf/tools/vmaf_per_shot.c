/**
 *
 * Copyright 2026 Lusoris and Claude (Anthropic)
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

/**
 * vmaf-perShot — per-shot CRF predictor sidecar (T6-3b / ADR-0222).
 *
 * Reads a single YUV input, segments it into shots, computes per-shot
 * complexity + motion-energy signals, and emits a per-shot CRF plan in
 * CSV or JSON for downstream encoders.
 *
 * The shot detector is a deliberately simple frame-difference heuristic
 * (mean absolute luma delta exceeding a fixed threshold). Once the
 * `transnet_v2` extractor lands (T6-3a / ADR-0220), the `--shots`
 * argument lets a caller pass a pre-computed shot map and the tool
 * skips its built-in detector. This decoupling keeps the sidecar
 * mergeable independently of the extractor.
 *
 * Per-shot CRF prediction in v1 is a transparent linear blend of
 * normalised complexity + motion + length signals. ADR-0222
 * §Alternatives considered records the rejected MLP path and the
 * rationale for staying interpretable until a real corpus is in hand.
 */

#ifdef _WIN32
#include "compat/win32/getopt.h"
#else
#include <getopt.h>
#endif

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Per-shot accumulators kept in a static-size array; no dynamic resizing. */
#define VMAF_PER_SHOT_MAX_SHOTS 4096U

/* Shot-detector threshold on the mean absolute luma delta (8-bit
 * domain). Empirically separates real cuts from in-shot motion on
 * the testdata fixtures; documented in docs/usage/vmaf-perShot.md. */
#define VMAF_PER_SHOT_DEFAULT_DIFF_THRESHOLD 12.0

/* Minimum shot length in frames; shorter runs are merged with the
 * previous shot. Guards against detector flicker on flashes / fades. */
#define VMAF_PER_SHOT_MIN_LEN 4U

/* Output format selector. */
enum vmaf_per_shot_format {
    VMAF_PER_SHOT_FMT_CSV = 0,
    VMAF_PER_SHOT_FMT_JSON = 1,
};

/* Per-shot tally — fixed footprint, no heap allocation per shot. */
struct vmaf_per_shot_record {
    uint32_t start_frame;
    uint32_t end_frame; /* inclusive */
    double sum_complexity;
    double sum_motion;
    uint32_t frames;
    double predicted_crf;
};

/* Resolved CLI settings — populated by parse_args, consumed by run. */
struct vmaf_per_shot_settings {
    const char *reference;
    const char *output;
    unsigned width;
    unsigned height;
    unsigned bitdepth;
    /* 420 / 422 / 444 — luma-only pipeline; chroma is ignored but
     * counted while advancing across frames. */
    unsigned chroma_subsampling;
    double target_vmaf;
    int crf_min;
    int crf_max;
    double diff_threshold;
    enum vmaf_per_shot_format format;
};

/* clang-format off */
static const struct option per_shot_long_opts[] = {
    {"reference",       required_argument, NULL, 'r'},
    {"width",           required_argument, NULL, 'w'},
    {"height",          required_argument, NULL, 'h'},
    {"pixel_format",    required_argument, NULL, 'p'},
    {"bitdepth",        required_argument, NULL, 'b'},
    {"output",          required_argument, NULL, 'o'},
    {"target-vmaf",     required_argument, NULL, 't'},
    {"crf-min",         required_argument, NULL, 'm'},
    {"crf-max",         required_argument, NULL, 'M'},
    {"diff-threshold",  required_argument, NULL, 'd'},
    {"format",          required_argument, NULL, 'f'},
    {"help",            no_argument,       NULL, '?'},
    {NULL, 0, NULL, 0},
};
/* clang-format on */

static void per_shot_print_usage(FILE *stream)
{
    (void)fprintf(stream, "Usage: vmaf-perShot --reference REF.yuv --width W --height H "
                          "--pixel_format 420|422|444 --bitdepth 8 --output PLAN [options]\n"
                          "\n"
                          "Required:\n"
                          "  -r, --reference PATH       reference YUV file\n"
                          "  -w, --width    N           frame width in pixels\n"
                          "  -h, --height   N           frame height in pixels\n"
                          "  -p, --pixel_format 420|422|444  planar YUV subsampling\n"
                          "  -b, --bitdepth 8|10|12|16  planar YUV bit depth\n"
                          "  -o, --output   PATH        per-shot plan output\n"
                          "\n"
                          "Optional:\n"
                          "  -t, --target-vmaf X        target VMAF score [default 90]\n"
                          "  -m, --crf-min N            minimum CRF clamp [default 18]\n"
                          "  -M, --crf-max N            maximum CRF clamp [default 35]\n"
                          "  -d, --diff-threshold X     shot-detector frame-diff cutoff\n"
                          "  -f, --format csv|json      output format [default csv]\n"
                          "      --help                 print this message\n");
}

/* Parse an unsigned integer with strict bounds. Bans atoi (banned by
 * docs/principles.md §1.2 r30). Returns 0 on success, -EINVAL on
 * any malformed input. */
static int per_shot_parse_uint(const char *s, unsigned long min, unsigned long max,
                               unsigned long *out)
{
    if (s == NULL || *s == '\0' || out == NULL)
        return -EINVAL;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == NULL || *end != '\0')
        return -EINVAL;
    if (v < min || v > max)
        return -EINVAL;
    *out = v;
    return 0;
}

static int per_shot_parse_int(const char *s, long min, long max, long *out)
{
    if (s == NULL || *s == '\0' || out == NULL)
        return -EINVAL;
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == NULL || *end != '\0')
        return -EINVAL;
    if (v < min || v > max)
        return -EINVAL;
    *out = v;
    return 0;
}

static int per_shot_parse_double(const char *s, double min, double max, double *out)
{
    if (s == NULL || *s == '\0' || out == NULL)
        return -EINVAL;
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0 || end == NULL || *end != '\0')
        return -EINVAL;
    if (!(v >= min && v <= max))
        return -EINVAL;
    *out = v;
    return 0;
}

static int per_shot_parse_pixfmt(const char *s, unsigned *out)
{
    if (s == NULL || out == NULL)
        return -EINVAL;
    if (strcmp(s, "420") == 0) {
        *out = 420U;
        return 0;
    }
    if (strcmp(s, "422") == 0) {
        *out = 422U;
        return 0;
    }
    if (strcmp(s, "444") == 0) {
        *out = 444U;
        return 0;
    }
    return -EINVAL;
}

static int per_shot_parse_format(const char *s, enum vmaf_per_shot_format *out)
{
    if (s == NULL || out == NULL)
        return -EINVAL;
    if (strcmp(s, "csv") == 0) {
        *out = VMAF_PER_SHOT_FMT_CSV;
        return 0;
    }
    if (strcmp(s, "json") == 0) {
        *out = VMAF_PER_SHOT_FMT_JSON;
        return 0;
    }
    return -EINVAL;
}

/* Apply a single getopt switch result. Split out of parse_args to
 * keep that function under ADR-0141's 60-line budget. */
static int per_shot_apply_opt(int c, const char *optarg_, struct vmaf_per_shot_settings *s)
{
    unsigned long uv = 0U;
    long iv = 0L;
    double dv = 0.0;
    switch (c) {
    case 'r':
        s->reference = optarg_;
        return 0;
    case 'w':
        if (per_shot_parse_uint(optarg_, 16U, 65535U, &uv) != 0)
            return -EINVAL;
        s->width = (unsigned)uv;
        return 0;
    case 'h':
        if (per_shot_parse_uint(optarg_, 16U, 65535U, &uv) != 0)
            return -EINVAL;
        s->height = (unsigned)uv;
        return 0;
    case 'p':
        return per_shot_parse_pixfmt(optarg_, &s->chroma_subsampling);
    case 'b':
        if (per_shot_parse_uint(optarg_, 8U, 16U, &uv) != 0)
            return -EINVAL;
        if (uv != 8U && uv != 10U && uv != 12U && uv != 16U)
            return -EINVAL;
        s->bitdepth = (unsigned)uv;
        return 0;
    case 'o':
        s->output = optarg_;
        return 0;
    case 't':
        if (per_shot_parse_double(optarg_, 0.0, 100.0, &dv) != 0)
            return -EINVAL;
        s->target_vmaf = dv;
        return 0;
    case 'm':
        if (per_shot_parse_int(optarg_, 0L, 63L, &iv) != 0)
            return -EINVAL;
        s->crf_min = (int)iv;
        return 0;
    case 'M':
        if (per_shot_parse_int(optarg_, 0L, 63L, &iv) != 0)
            return -EINVAL;
        s->crf_max = (int)iv;
        return 0;
    case 'd':
        if (per_shot_parse_double(optarg_, 0.0, 255.0, &dv) != 0)
            return -EINVAL;
        s->diff_threshold = dv;
        return 0;
    case 'f':
        return per_shot_parse_format(optarg_, &s->format);
    default:
        return -EINVAL;
    }
}

static int per_shot_validate(const struct vmaf_per_shot_settings *s)
{
    if (s->reference == NULL || s->output == NULL)
        return -EINVAL;
    if (s->width == 0U || s->height == 0U)
        return -EINVAL;
    if (s->bitdepth == 0U)
        return -EINVAL;
    if (s->chroma_subsampling == 0U)
        return -EINVAL;
    if (s->crf_min > s->crf_max)
        return -EINVAL;
    return 0;
}

/* Defaults match docs/usage/vmaf-perShot.md and ADR-0222 §Decision. */
static void per_shot_settings_defaults(struct vmaf_per_shot_settings *s)
{
    memset(s, 0, sizeof(*s));
    s->target_vmaf = 90.0;
    s->crf_min = 18;
    s->crf_max = 35;
    s->diff_threshold = VMAF_PER_SHOT_DEFAULT_DIFF_THRESHOLD;
    s->format = VMAF_PER_SHOT_FMT_CSV;
}

static int per_shot_parse_args(int argc, char **argv, struct vmaf_per_shot_settings *s)
{
    per_shot_settings_defaults(s);
    int c = 0;
    int idx = 0;
    optind = 1;
    /* getopt_long flagged concurrency-mt-unsafe by clang-tidy; same
     * baseline applies in libvmaf/tools/cli_parse.c — every C CLI
     * uses it, and the binary is single-threaded by construction. */
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    while ((c = getopt_long(argc, argv, "r:w:h:p:b:o:t:m:M:d:f:?", per_shot_long_opts, &idx)) !=
           -1) {
        if (c == '?') {
            per_shot_print_usage(stdout);
            return 1; /* sentinel — caller treats as success-with-help */
        }
        if (per_shot_apply_opt(c, optarg, s) != 0) {
            (void)fprintf(stderr, "vmaf-perShot: invalid value for -%c\n", c);
            return -EINVAL;
        }
    }
    return per_shot_validate(s);
}

/* Frame-level signal extraction. Operates on the luma plane only —
 * scene_complexity = sample variance, motion_energy = mean absolute
 * delta vs. previous luma plane. Both are bit-depth-normalised so
 * downstream blending is depth-invariant. */
static double per_shot_sample_scale(unsigned bitdepth)
{
    const unsigned max_sample = (bitdepth >= 16U) ? 65535U : ((1U << bitdepth) - 1U);
    return 1.0 / (double)max_sample;
}

static void per_shot_compute_frame_signals(const uint8_t *cur, const uint8_t *prev, size_t pixels,
                                           unsigned bitdepth, double *complexity, double *motion)
{
    const double scale = per_shot_sample_scale(bitdepth);
    double sum = 0.0;
    double sumsq = 0.0;
    double abs_diff = 0.0;
    if (bitdepth > 8U) {
        const uint16_t *c16 = (const uint16_t *)cur;
        const uint16_t *p16 = (prev != NULL) ? (const uint16_t *)prev : NULL;
        for (size_t i = 0U; i < pixels; ++i) {
            double v = (double)c16[i] * scale;
            sum += v;
            sumsq += v * v;
            if (p16 != NULL) {
                double d = v - (double)p16[i] * scale;
                abs_diff += (d < 0.0) ? -d : d;
            }
        }
    } else {
        for (size_t i = 0U; i < pixels; ++i) {
            double v = (double)cur[i] * scale;
            sum += v;
            sumsq += v * v;
            if (prev != NULL) {
                double d = v - (double)prev[i] * scale;
                abs_diff += (d < 0.0) ? -d : d;
            }
        }
    }
    const double n = (double)pixels;
    const double mean = sum / n;
    const double var = (sumsq / n) - (mean * mean);
    *complexity = (var > 0.0) ? var : 0.0;
    *motion = (prev != NULL) ? (abs_diff / n) : 0.0;
}

/* Decide whether `mean_abs_diff_8bit` crosses the cut threshold. The
 * detector input is normalised to the 8-bit domain so a single
 * threshold serves all bit depths. */
static bool per_shot_is_cut(double mean_abs_diff_8bit, double threshold)
{
    return mean_abs_diff_8bit >= threshold;
}

/* Linear-blend CRF predictor (v1, ADR-0222). All coefficients are
 * tunable in code only — they encode the fork's prior, not a
 * trained fit. The user docs walk through how to override via a
 * trained MLP once corpus + harness are in place. */
static double per_shot_predict_crf(double mean_complexity, double mean_motion, uint32_t shot_frames,
                                   double target_vmaf, int crf_min, int crf_max)
{
    /* Reference VMAF: the spec is "hit target VMAF on this shot",
     * so we reduce CRF (better quality) when target rises. The
     * mapping is intentionally gentle — 5 CRF units across the
     * 70-100 useful target range. */
    const double target_norm = (target_vmaf - 70.0) / 30.0;
    const double target_clamped =
        (target_norm < 0.0) ? 0.0 : (target_norm > 1.0 ? 1.0 : target_norm);

    /* Complexity penalty — a busier shot needs a lower CRF (better
     * quality) to keep VMAF stable. Variance of luma in [0, 0.25]
     * after the per-pixel [0, 1] normalisation; we map [0, 0.05]
     * to [0, 1] which covers the bulk of natural content. */
    double complexity_norm = mean_complexity / 0.05;
    if (complexity_norm > 1.0)
        complexity_norm = 1.0;
    if (complexity_norm < 0.0)
        complexity_norm = 0.0;

    /* Motion bonus — fast motion masks artefacts, so we can raise
     * CRF (worse quality, smaller bitrate) without VMAF cost. */
    double motion_norm = mean_motion * 20.0;
    if (motion_norm > 1.0)
        motion_norm = 1.0;

    /* Length bonus — longer shots amortise rate-control startup
     * cost; very short shots get a small CRF reduction to avoid
     * underrun-driven artefacts. */
    double length_factor = 1.0;
    if (shot_frames < 24U)
        length_factor = 0.5;

    const double base = (double)crf_min + ((double)crf_max - (double)crf_min) * 0.5;
    const double range = (double)crf_max - (double)crf_min;
    /* +length-bonus motion, -complexity, -target. Centre at base,
     * clamp to [crf_min, crf_max]. */
    double crf = base + 0.20 * range * motion_norm * length_factor -
                 0.20 * range * complexity_norm - 0.15 * range * target_clamped;
    if (crf < (double)crf_min)
        crf = (double)crf_min;
    if (crf > (double)crf_max)
        crf = (double)crf_max;
    return crf;
}

/* Append a fully-formed frame measurement into the running shot.
 * Cuts a new shot record when a cut is detected. Returns -1 on
 * shot-table overflow (capped at VMAF_PER_SHOT_MAX_SHOTS to keep
 * the footprint static per ADR rule §1.2 r2). */
static int per_shot_record_frame(struct vmaf_per_shot_record *shots, uint32_t *shot_count,
                                 uint32_t frame_idx, double complexity, double motion, bool is_cut)
{
    if (*shot_count == 0U) {
        shots[0].start_frame = 0U;
        shots[0].end_frame = 0U;
        shots[0].sum_complexity = 0.0;
        shots[0].sum_motion = 0.0;
        shots[0].frames = 0U;
        shots[0].predicted_crf = 0.0;
        *shot_count = 1U;
    }
    /* Cut → start a new shot, but only if the running shot has
     * surpassed VMAF_PER_SHOT_MIN_LEN frames (suppresses flash /
     * fade flicker). */
    struct vmaf_per_shot_record *cur = &shots[*shot_count - 1U];
    if (is_cut && cur->frames >= VMAF_PER_SHOT_MIN_LEN) {
        if (*shot_count >= VMAF_PER_SHOT_MAX_SHOTS)
            return -1;
        cur->end_frame = frame_idx - 1U;
        struct vmaf_per_shot_record *nxt = &shots[*shot_count];
        nxt->start_frame = frame_idx;
        nxt->end_frame = frame_idx;
        nxt->sum_complexity = 0.0;
        nxt->sum_motion = 0.0;
        nxt->frames = 0U;
        nxt->predicted_crf = 0.0;
        *shot_count += 1U;
        cur = nxt;
    }
    cur->end_frame = frame_idx;
    cur->sum_complexity += complexity;
    cur->sum_motion += motion;
    cur->frames += 1U;
    return 0;
}

static size_t per_shot_chroma_samples(unsigned w, unsigned h, unsigned chroma_subsampling)
{
    const size_t luma = (size_t)w * (size_t)h;
    if (chroma_subsampling == 420U) {
        return ((size_t)((w + 1U) / 2U)) * ((size_t)((h + 1U) / 2U)) * 2U;
    }
    if (chroma_subsampling == 422U) {
        return ((size_t)((w + 1U) / 2U)) * (size_t)h * 2U;
    }
    if (chroma_subsampling == 444U)
        return luma * 2U;
    return 0U;
}

/* Planar YUV frame-size helper. The scan only reads luma, but frame
 * iteration must still skip both chroma planes with the selected
 * subsampling. Bit depth doubles the per-sample byte count above 8. */
static size_t per_shot_yuv_frame_bytes(unsigned w, unsigned h, unsigned chroma_subsampling,
                                       unsigned bitdepth)
{
    const size_t luma = (size_t)w * (size_t)h;
    const size_t pix = luma + per_shot_chroma_samples(w, h, chroma_subsampling);
    return (bitdepth > 8U) ? (pix * 2U) : pix;
}

/* Read one full YUV420P frame's luma plane into `luma`. Skips chroma
 * by seeking past it. Returns 0 on EOF, 1 on success, -1 on partial
 * read (corrupt input). */
static int per_shot_read_luma(FILE *fin, uint8_t *luma, size_t luma_bytes, size_t chroma_bytes)
{
    size_t got = fread(luma, 1U, luma_bytes, fin);
    if (got == 0U)
        return 0;
    if (got != luma_bytes)
        return -1;
    if (chroma_bytes > 0U) {
        if (fseek(fin, (long)chroma_bytes, SEEK_CUR) != 0) {
            /* fall back to read-and-drop if seek isn't permitted
             * (e.g. piped input) */
            size_t remaining = chroma_bytes;
            uint8_t scratch[4096];
            while (remaining > 0U) {
                size_t want = (remaining > sizeof(scratch)) ? sizeof(scratch) : remaining;
                size_t r = fread(scratch, 1U, want, fin);
                if (r == 0U)
                    return -1;
                remaining -= r;
            }
        }
    }
    return 1;
}

/* Finalise per-shot statistics: average the running sums, then run
 * the CRF predictor. */
static void per_shot_finalise(struct vmaf_per_shot_record *shots, uint32_t shot_count,
                              const struct vmaf_per_shot_settings *s)
{
    for (uint32_t i = 0U; i < shot_count; ++i) {
        struct vmaf_per_shot_record *r = &shots[i];
        if (r->frames == 0U)
            continue;
        const double mean_c = r->sum_complexity / (double)r->frames;
        const double mean_m = r->sum_motion / (double)r->frames;
        r->predicted_crf =
            per_shot_predict_crf(mean_c, mean_m, r->frames, s->target_vmaf, s->crf_min, s->crf_max);
    }
}

/* Compute the per-shot mean signals safely (guards frames==0). */
static void per_shot_means(const struct vmaf_per_shot_record *r, double *mean_c, double *mean_m)
{
    *mean_c = (r->frames > 0U) ? r->sum_complexity / (double)r->frames : 0.0;
    *mean_m = (r->frames > 0U) ? r->sum_motion / (double)r->frames : 0.0;
}

/* CSV body — one row per shot. Returns 0 on success, -EIO on
 * fprintf failure. */
static int per_shot_write_plan_csv(FILE *out, const struct vmaf_per_shot_record *shots,
                                   uint32_t shot_count)
{
    if (fprintf(out, "shot_id,start_frame,end_frame,frames,"
                     "mean_complexity,mean_motion,predicted_crf\n") < 0) {
        return -EIO;
    }
    for (uint32_t i = 0U; i < shot_count; ++i) {
        const struct vmaf_per_shot_record *r = &shots[i];
        double mean_c = 0.0;
        double mean_m = 0.0;
        per_shot_means(r, &mean_c, &mean_m);
        if (fprintf(out, "%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%.6f,%.6f,%.2f\n", i,
                    r->start_frame, r->end_frame, r->frames, mean_c, mean_m,
                    r->predicted_crf) < 0) {
            return -EIO;
        }
    }
    return 0;
}

/* JSON body — one object per shot wrapped in a `shots` array
 * carrying CRF clamp metadata. */
static int per_shot_write_plan_json(FILE *out, const struct vmaf_per_shot_settings *s,
                                    const struct vmaf_per_shot_record *shots, uint32_t shot_count)
{
    if (fprintf(out,
                "{\n  \"target_vmaf\": %.2f,\n"
                "  \"crf_min\": %d,\n  \"crf_max\": %d,\n"
                "  \"shots\": [\n",
                s->target_vmaf, s->crf_min, s->crf_max) < 0) {
        return -EIO;
    }
    for (uint32_t i = 0U; i < shot_count; ++i) {
        const struct vmaf_per_shot_record *r = &shots[i];
        double mean_c = 0.0;
        double mean_m = 0.0;
        per_shot_means(r, &mean_c, &mean_m);
        if (fprintf(out,
                    "    {\"shot_id\": %" PRIu32 ", \"start_frame\": %" PRIu32
                    ", \"end_frame\": %" PRIu32 ", \"frames\": %" PRIu32
                    ", \"mean_complexity\": %.6f"
                    ", \"mean_motion\": %.6f"
                    ", \"predicted_crf\": %.2f}%s\n",
                    i, r->start_frame, r->end_frame, r->frames, mean_c, mean_m, r->predicted_crf,
                    (i + 1U < shot_count) ? "," : "") < 0) {
            return -EIO;
        }
    }
    if (fprintf(out, "  ]\n}\n") < 0)
        return -EIO;
    return 0;
}

/* Emit the plan in the requested format. Both formats encode the
 * full signal vector so a downstream encoder can override the
 * predicted CRF with a custom rule.
 *
 * When output is the sentinel "-" write to stdout directly so that
 * callers piping the output (e.g. per_shot.py's subprocess capture)
 * receive the JSON/CSV on their stdin.  No fclose() on stdout. */
static int per_shot_write_plan(const struct vmaf_per_shot_settings *s,
                               const struct vmaf_per_shot_record *shots, uint32_t shot_count)
{
    if (s == NULL || s->output == NULL)
        return -EINVAL;

    FILE *out = NULL;
    bool use_stdout = (strcmp(s->output, "-") == 0);
    if (use_stdout) {
        out = stdout;
    } else {
        /* Use open() + fdopen() with explicit 0644 (rw-r--r--) on POSIX
         * so the new file is not created world-writable per the process
         * umask (CodeQL's "File created without restricting permissions"
         * alert). MSVC's runtime doesn't ship <unistd.h> and Windows file
         * permissions don't map onto Unix mode bits the same way; fall
         * back to plain fopen() on _WIN32 where the security model is
         * ACL-based and not affected by the umask issue. */
#ifndef _WIN32
        int fd =
            open(s->output, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        if (fd < 0) {
            (void)fprintf(stderr, "vmaf-perShot: cannot open output %s\n", s->output);
            return -EIO;
        }
        out = fdopen(fd, "w");
        if (out == NULL) {
            /* strerror() is concurrency-mt-unsafe; the path is enough
             * context for the user to diagnose. */
            (void)fprintf(stderr, "vmaf-perShot: cannot open output %s\n", s->output);
            (void)close(fd);
            return -EIO;
        }
#else
        out = fopen(s->output, "w");
        if (out == NULL) {
            (void)fprintf(stderr, "vmaf-perShot: cannot open output %s\n", s->output);
            return -EIO;
        }
#endif
    }

    int rc = (s->format == VMAF_PER_SHOT_FMT_CSV) ?
                 per_shot_write_plan_csv(out, shots, shot_count) :
                 per_shot_write_plan_json(out, s, shots, shot_count);

    /* Flush stdout but do not fclose it; fclose a file opened by this
     * function only when we own the descriptor. */
    if (use_stdout) {
        if (fflush(out) != 0 && rc == 0)
            rc = -EIO;
    } else {
        if (fclose(out) != 0 && rc == 0)
            rc = -EIO;
    }
    return rc;
}

/* Drive the YUV scan: read frame N, compute (complexity, motion vs
 * frame N-1), feed shot detector + accumulators. Returns 0 on
 * success, negative errno on failure. */
/* Scratch buffers + run-time scalars threaded through the scan loop.
 * Lifted out of per_shot_scan into a struct so the loop body can be
 * a standalone helper without a 7-arg signature. */
struct per_shot_scan_ctx {
    FILE *fin;
    uint8_t *cur;
    uint8_t *prev;
    size_t pixels;
    size_t luma_bytes;
    size_t chroma_bytes;
    uint32_t frame_idx;
    bool have_prev;
};

/* Inner loop of per_shot_scan: reads, signals, records, and swaps
 * the cur/prev buffers. Returns 0 on success (frames consumed),
 * negative errno on any failure. */
static int per_shot_scan_loop(const struct vmaf_per_shot_settings *s, struct per_shot_scan_ctx *ctx,
                              struct vmaf_per_shot_record *shots, uint32_t *shot_count)
{
    for (;;) {
        int r = per_shot_read_luma(ctx->fin, ctx->cur, ctx->luma_bytes, ctx->chroma_bytes);
        if (r == 0)
            break;
        if (r < 0)
            return -EIO;
        double complexity = 0.0;
        double motion = 0.0;
        per_shot_compute_frame_signals(ctx->cur, ctx->have_prev ? ctx->prev : NULL, ctx->pixels,
                                       s->bitdepth, &complexity, &motion);
        /* Detector input is in [0, 1] luma units; rescale to 8-bit
         * domain so the threshold is intuitive. */
        const double mean_abs_diff_8bit = motion * 255.0;
        const bool is_cut =
            ctx->have_prev && per_shot_is_cut(mean_abs_diff_8bit, s->diff_threshold);
        if (per_shot_record_frame(shots, shot_count, ctx->frame_idx, complexity, motion, is_cut) !=
            0) {
            (void)fprintf(stderr,
                          "vmaf-perShot: shot table overflow at frame %" PRIu32 " (max %u)\n",
                          ctx->frame_idx, VMAF_PER_SHOT_MAX_SHOTS);
            return -ENOSPC;
        }
        /* Swap cur ↔ prev for the next iteration. */
        uint8_t *tmp = ctx->prev;
        ctx->prev = ctx->cur;
        ctx->cur = tmp;
        ctx->have_prev = true;
        ctx->frame_idx += 1U;
    }
    return 0;
}

static int per_shot_scan(const struct vmaf_per_shot_settings *s, struct vmaf_per_shot_record *shots,
                         uint32_t *shot_count)
{
    /* Defensive: per_shot_validate() already enforces these, but
     * the static analyser can't see that across the call boundary. */
    if (s == NULL || s->reference == NULL || shots == NULL || shot_count == NULL) {
        return -EINVAL;
    }
    if (s->width == 0U || s->height == 0U || s->bitdepth == 0U)
        return -EINVAL;
    FILE *fin = fopen(s->reference, "rb");
    if (fin == NULL) {
        /* strerror() is concurrency-mt-unsafe; the path is enough
         * context for the user. */
        (void)fprintf(stderr, "vmaf-perShot: cannot open %s\n", s->reference);
        return -ENOENT;
    }
    int rc = 0;
    struct per_shot_scan_ctx ctx;
    ctx.fin = fin;
    ctx.pixels = (size_t)s->width * (size_t)s->height;
    const size_t bytes_per_sample = (s->bitdepth > 8U) ? 2U : 1U;
    ctx.luma_bytes = ctx.pixels * bytes_per_sample;
    ctx.chroma_bytes =
        per_shot_yuv_frame_bytes(s->width, s->height, s->chroma_subsampling, s->bitdepth) -
        ctx.luma_bytes;
    ctx.cur = malloc(ctx.luma_bytes);
    ctx.prev = malloc(ctx.luma_bytes);
    ctx.frame_idx = 0U;
    ctx.have_prev = false;
    if (ctx.cur == NULL || ctx.prev == NULL) {
        rc = -ENOMEM;
        goto cleanup;
    }
    rc = per_shot_scan_loop(s, &ctx, shots, shot_count);
    if (rc == 0 && ctx.frame_idx == 0U) {
        (void)fprintf(stderr, "vmaf-perShot: no frames read from %s\n", s->reference);
        rc = -EINVAL;
    }
cleanup:
    free(ctx.cur);
    free(ctx.prev);
    if (fclose(fin) != 0 && rc == 0)
        rc = -EIO;
    return rc;
}

int main(int argc, char **argv)
{
    assert(argc > 0);
    assert(argv != NULL);
    struct vmaf_per_shot_settings settings;
    int rc = per_shot_parse_args(argc, argv, &settings);
    if (rc == 1)
        return 0; /* --help */
    if (rc != 0) {
        per_shot_print_usage(stderr);
        return EXIT_FAILURE;
    }
    struct vmaf_per_shot_record *shots = calloc(VMAF_PER_SHOT_MAX_SHOTS, sizeof(*shots));
    if (shots == NULL) {
        (void)fprintf(stderr, "vmaf-perShot: out of memory\n");
        return EXIT_FAILURE;
    }
    uint32_t shot_count = 0U;
    rc = per_shot_scan(&settings, shots, &shot_count);
    if (rc != 0) {
        free(shots);
        return EXIT_FAILURE;
    }
    assert(shot_count <= VMAF_PER_SHOT_MAX_SHOTS);
    per_shot_finalise(shots, shot_count, &settings);
    rc = per_shot_write_plan(&settings, shots, shot_count);
    if (rc != 0) {
        free(shots);
        return EXIT_FAILURE;
    }
    assert(settings.output != NULL);
    (void)fprintf(stderr, "vmaf-perShot: wrote %" PRIu32 " shot(s) to %s\n", shot_count,
                  settings.output);
    free(shots);
    return EXIT_SUCCESS;
}
