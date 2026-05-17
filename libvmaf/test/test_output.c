/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
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

/*
 * Coverage gap follow-up (R3 from docs/development/coverage-gap-analysis-2026-05-02.md):
 * exercise libvmaf/src/output.c writer-format paths.
 *
 * The four writers (XML, JSON, CSV, SUB) are reached end-to-end only by full
 * VMAF runs — pre-this test the CPU coverage gate measured output.c at 28%
 * because only the JSON path was indirectly exercised. This test drives each
 * writer through tmpfile-backed FILE* sinks with synthetic VmafFeatureCollector
 * state, asserting the output is non-empty and contains expected markers
 * (frame indices, feature names, format-specific structural tokens).
 *
 * Branches exercised that were dead pre-this test:
 *   - CSV header + data rows (vmaf_write_output_csv).
 *   - SUB per-frame line (vmaf_write_output_sub).
 *   - XML EINVAL guards (NULL vmaf / fc / outfile).
 *   - NaN / +Inf -> "null" emission in JSON frame metrics, JSON pooled
 *     scores, JSON aggregate metrics, and JSON top-level fps.
 *   - subsample > 1 frame skipping.
 *   - count_written_at == 0 skip path (frames where no feature wrote).
 *   - score_format=NULL fall-through to DEFAULT_SCORE_FORMAT.
 *   - Custom score_format ("%.3f") overrides default.
 *   - aggregate_metrics with multiple entries (comma separators).
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "feature/feature_collector.h"
#include "libvmaf/libvmaf.h"

/* Pull output.c + libvmaf.c in directly so (a) the .gcno/.gcda sit in this
 * test's build dir and gcovr aggregates them into the coverage report, and
 * (b) we get the full `struct VmafContext` definition (the public header
 * exposes only an opaque typedef). Mirrors the test_feature_collector
 * pattern of including .c files. */
#include "libvmaf.c"
#include "output.c"

/* Read entire FILE* (after rewind) into a malloc'd, NUL-terminated buffer.
 * Caller frees. Returns NULL on failure. Bounded scan: 64 KiB cap is plenty
 * for our synthetic collectors (largest test emits ~2 KiB). */
static char *slurp(FILE *f)
{
    if (fseek(f, 0, SEEK_END) != 0)
        return NULL;
    long sz = ftell(f);
    if (sz < 0 || sz > (1L << 16))
        return NULL;
    if (fseek(f, 0, SEEK_SET) != 0)
        return NULL;
    char *buf = malloc((size_t)sz + 1);
    if (!buf)
        return NULL;
    size_t n = fread(buf, 1, (size_t)sz, f);
    buf[n] = '\0';
    return buf;
}

/* Append `score` for `name` at `index` to the collector. mu_assert on
 * failure. */
static int seed(VmafFeatureCollector *fc, const char *name, double score, unsigned index)
{
    return vmaf_feature_collector_append(fc, name, score, index);
}

/* Centralised teardown — keeps each test body short enough to stay under
 * `readability-function-size` and silences cert-err33-c on the void-return
 * cleanup calls (fclose / free have no recoverable error here, vmaf_close
 * is best-effort during teardown). */
static void teardown(VmafContext *vmaf, FILE *f, char *out)
{
    free(out);
    if (f)
        (void)fclose(f);
    if (vmaf)
        (void)vmaf_close(vmaf);
}

/* Build a VmafContext + populate its feature_collector with two features
 * across three frames; one frame deliberately left empty to exercise the
 * count_written_at == 0 skip path. */
static int seed_normal(VmafContext **out_vmaf)
{
    VmafConfiguration cfg = {0};
    int err = vmaf_init(out_vmaf, cfg);
    if (err)
        return err;
    VmafFeatureCollector *fc = (*out_vmaf)->feature_collector;
    /* frame 0: both features written */
    err |= seed(fc, "feat_a", 80.0, 0);
    err |= seed(fc, "feat_b", 0.5, 0);
    /* frame 1: only feat_a — exercises the per-feature `written` skip
     * inside the per-row loops. */
    err |= seed(fc, "feat_a", 81.0, 1);
    /* frame 2: nothing written — skipped by count_written_at == 0. */
    /* frame 3: both written. */
    err |= seed(fc, "feat_a", 82.0, 3);
    err |= seed(fc, "feat_b", 0.6, 3);
    return err;
}

static char *test_csv_basic()
{
    VmafContext *vmaf;
    int err = seed_normal(&vmaf);
    mu_assert("seed_normal failed", !err);

    FILE *f = tmpfile();
    mu_assert("tmpfile failed", f);

    err = vmaf_write_output_csv(vmaf->feature_collector, f, /*subsample=*/0,
                                /*score_format=*/NULL);
    mu_assert("vmaf_write_output_csv returned non-zero", !err);

    char *out = slurp(f);
    mu_assert("slurp failed", out);

    /* Header row contains both feature names. */
    mu_assert("csv: missing 'Frame,' header", strstr(out, "Frame,"));
    mu_assert("csv: missing feat_a in header", strstr(out, "feat_a,"));
    mu_assert("csv: missing feat_b in header", strstr(out, "feat_b,"));
    /* Data row for frame 0 emits both scores. */
    mu_assert("csv: missing 80.000000 (feat_a frame 0)", strstr(out, "80.000000"));
    mu_assert("csv: missing 0.500000 (feat_b frame 0)", strstr(out, "0.500000"));
    /* Frame 2 not present — count_written_at skipped it. */
    mu_assert("csv: frame 2 should not appear (no scores written)", !strstr(out, "\n2,"));
    /* Frame 3 present. */
    mu_assert("csv: frame 3 missing", strstr(out, "\n3,"));

    teardown(vmaf, f, out);
    return NULL;
}

static char *test_csv_subsample_and_custom_format()
{
    VmafContext *vmaf;
    int err = seed_normal(&vmaf);
    mu_assert("seed_normal failed", !err);

    FILE *f = tmpfile();
    mu_assert("tmpfile failed", f);

    /* subsample=2 keeps even-indexed frames (0, 2, ...). Frame 1 is dropped
     * by the modulo guard; frame 3 is dropped too (3 % 2 != 0). Frame 0 is
     * the only survivor. Custom format "%.3f" overrides DEFAULT_SCORE_FORMAT. */
    err = vmaf_write_output_csv(vmaf->feature_collector, f, /*subsample=*/2, "%.3f");
    mu_assert("vmaf_write_output_csv returned non-zero", !err);

    char *out = slurp(f);
    mu_assert("slurp failed", out);

    /* "%.3f" of 80.0 == "80.000". The 6-digit default would print "80.000000";
     * we want to make sure the custom format won. */
    mu_assert("csv: custom format missing '80.000'", strstr(out, "80.000"));
    mu_assert("csv: custom format must not produce default 80.000000", !strstr(out, "80.000000"));
    /* Frame 1 dropped by subsample. */
    mu_assert("csv: subsample=2 should drop frame 1", !strstr(out, "\n1,"));
    /* Frame 3 dropped by subsample (odd). */
    mu_assert("csv: subsample=2 should drop frame 3", !strstr(out, "\n3,"));

    teardown(vmaf, f, out);
    return NULL;
}

static char *test_sub_basic()
{
    VmafContext *vmaf;
    int err = seed_normal(&vmaf);
    mu_assert("seed_normal failed", !err);

    FILE *f = tmpfile();
    mu_assert("tmpfile failed", f);

    err = vmaf_write_output_sub(vmaf->feature_collector, f, /*subsample=*/0,
                                /*score_format=*/NULL);
    mu_assert("vmaf_write_output_sub returned non-zero", !err);

    char *out = slurp(f);
    mu_assert("slurp failed", out);

    /* SUB format prefixes each line with "{N}{N+1}frame: N|" then per-feature
     * "name: value|" entries. Verify frame 0 and frame 1 are emitted; frame 2
     * skipped; frame 3 present. */
    mu_assert("sub: missing frame 0 marker", strstr(out, "{0}{1}frame: 0|"));
    mu_assert("sub: missing frame 1 marker", strstr(out, "{1}{2}frame: 1|"));
    mu_assert("sub: frame 2 must be skipped (no scores)", !strstr(out, "{2}{3}frame: 2|"));
    mu_assert("sub: missing frame 3 marker", strstr(out, "{3}{4}frame: 3|"));
    mu_assert("sub: missing feat_a:", strstr(out, "feat_a: "));
    mu_assert("sub: missing feat_b:", strstr(out, "feat_b: "));

    teardown(vmaf, f, out);
    return NULL;
}

static char *test_xml_einval_guards()
{
    /* All three guards live at the head of vmaf_write_output_xml. */
    VmafContext *vmaf;
    int err = seed_normal(&vmaf);
    mu_assert("seed_normal failed", !err);

    FILE *f = tmpfile();
    mu_assert("tmpfile failed", f);

    err = vmaf_write_output_xml(NULL, vmaf->feature_collector, f, 0, 64, 64, 30.0, 1, NULL);
    mu_assert("xml: NULL vmaf must return -EINVAL", err == -EINVAL);

    err = vmaf_write_output_xml(vmaf, NULL, f, 0, 64, 64, 30.0, 1, NULL);
    mu_assert("xml: NULL fc must return -EINVAL", err == -EINVAL);

    err = vmaf_write_output_xml(vmaf, vmaf->feature_collector, NULL, 0, 64, 64, 30.0, 1, NULL);
    mu_assert("xml: NULL outfile must return -EINVAL", err == -EINVAL);

    teardown(vmaf, f, NULL);
    return NULL;
}

static char *test_xml_basic()
{
    /* For the pooled-metrics block to emit per-feature mean/min/max/harmonic
     * entries, *every* frame in [0, pic_cnt) must have a written value for
     * each feature; vmaf_feature_score_pooled bails on the first missing
     * index. Use a dense 2-frame x 2-feature collector here. The
     * count_written_at skip branch is already covered by test_csv_basic /
     * test_sub_basic (which use seed_normal). */
    VmafContext *vmaf;
    VmafConfiguration cfg = {0};
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    VmafFeatureCollector *fc = vmaf->feature_collector;
    err |= seed(fc, "feat_a", 80.0, 0);
    err |= seed(fc, "feat_a", 82.0, 1);
    err |= seed(fc, "feat_b", 0.5, 0);
    err |= seed(fc, "feat_b", 0.6, 1);
    mu_assert("seed dense failed", !err);

    /* Aggregate exercises the aggregate_metrics block. */
    err = vmaf_feature_collector_set_aggregate(fc, "agg_x", 7.25);
    mu_assert("set_aggregate failed", !err);

    FILE *f = tmpfile();
    mu_assert("tmpfile failed", f);

    err = vmaf_write_output_xml(vmaf, fc, f, /*subsample=*/0,
                                /*width=*/640, /*height=*/360, /*fps=*/29.97,
                                /*pic_cnt=*/2, /*score_format=*/NULL);
    mu_assert("xml: writer returned non-zero", !err);

    char *out = slurp(f);
    mu_assert("slurp failed", out);

    mu_assert("xml: missing root open <VMAF version=", strstr(out, "<VMAF version="));
    mu_assert("xml: missing root close </VMAF>", strstr(out, "</VMAF>"));
    mu_assert("xml: missing <params qualityWidth=\"640\"",
              strstr(out, "<params qualityWidth=\"640\""));
    mu_assert("xml: missing <fyi fps=\"29.97\"", strstr(out, "<fyi fps=\"29.97\""));
    mu_assert("xml: missing <frames>", strstr(out, "<frames>"));
    mu_assert("xml: missing </frames>", strstr(out, "</frames>"));
    mu_assert("xml: missing <pooled_metrics>", strstr(out, "<pooled_metrics>"));
    mu_assert("xml: missing </pooled_metrics>", strstr(out, "</pooled_metrics>"));
    mu_assert("xml: missing <aggregate_metrics", strstr(out, "<aggregate_metrics "));
    mu_assert("xml: missing agg_x in aggregate_metrics", strstr(out, "agg_x=\""));
    mu_assert("xml: missing pooled mean for feat_a", strstr(out, "mean=\""));

    teardown(vmaf, f, out);
    return NULL;
}

static char *test_json_basic_and_format()
{
    /* Dense 2x2 collector so json_write_pooled_entry / json_write_pool_score
     * produce per-method numbers (otherwise vmaf_feature_score_pooled
     * returns -EAGAIN and the writer emits an empty per-feature block). */
    VmafContext *vmaf;
    VmafConfiguration cfg = {0};
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    VmafFeatureCollector *fc = vmaf->feature_collector;
    err |= seed(fc, "feat_a", 80.0, 0);
    err |= seed(fc, "feat_a", 82.0, 1);
    err |= seed(fc, "feat_b", 0.5, 0);
    err |= seed(fc, "feat_b", 0.6, 1);
    mu_assert("seed dense failed", !err);

    err = vmaf_feature_collector_set_aggregate(fc, "agg_x", 7.25);
    mu_assert("set_aggregate failed", !err);
    err = vmaf_feature_collector_set_aggregate(fc, "agg_y", 1.5);
    mu_assert("set_aggregate failed", !err);

    FILE *f = tmpfile();
    mu_assert("tmpfile failed", f);

    /* Use "%.17g" — the round-trip lossless ADR-0119 format. */
    err = vmaf_write_output_json(vmaf, fc, f, /*subsample=*/0,
                                 /*fps=*/24.0, /*pic_cnt=*/2, "%.17g");
    mu_assert("json: writer returned non-zero", !err);

    char *out = slurp(f);
    mu_assert("slurp failed", out);

    /* Top-level scalar fields. */
    mu_assert("json: missing \"version\"", strstr(out, "\"version\":"));
    mu_assert("json: missing \"fps\": 24.00", strstr(out, "\"fps\": 24.00"));
    /* Frames array, pooled_metrics block, aggregate_metrics block. */
    mu_assert("json: missing \"frames\":", strstr(out, "\"frames\":"));
    mu_assert("json: missing \"frameNum\"", strstr(out, "\"frameNum\":"));
    mu_assert("json: missing \"metrics\":", strstr(out, "\"metrics\":"));
    mu_assert("json: missing \"pooled_metrics\":", strstr(out, "\"pooled_metrics\":"));
    /* Dense collector + pic_cnt=2 means every pool method emits a number. */
    mu_assert("json: missing \"mean\" inside pooled_metrics", strstr(out, "\"mean\":"));
    mu_assert("json: missing \"min\" inside pooled_metrics", strstr(out, "\"min\":"));
    mu_assert("json: missing \"max\" inside pooled_metrics", strstr(out, "\"max\":"));
    mu_assert("json: missing \"harmonic_mean\" inside pooled_metrics",
              strstr(out, "\"harmonic_mean\":"));
    mu_assert("json: missing \"aggregate_metrics\":", strstr(out, "\"aggregate_metrics\":"));
    /* Two aggregates -> at least one comma separator inside the block. */
    mu_assert("json: missing agg_x", strstr(out, "\"agg_x\":"));
    mu_assert("json: missing agg_y", strstr(out, "\"agg_y\":"));
    /* "%.17g" formats 80.0 as "80" (no fractional zeros), distinguishing it
     * from the default "%.6f" which would render "80.000000". */
    mu_assert("json: %.17g should produce \"80,\" or \"80\\n\" not \"80.000000\"",
              !strstr(out, "80.000000"));

    teardown(vmaf, f, out);
    return NULL;
}

static char *test_json_nan_and_inf()
{
    /* Force NaN / +Inf into both frame metrics, pooled (via mean over the
     * frame values), and aggregates / fps. The writers route every numeric
     * branch through fpclassify(); NaN + Inf must serialize as JSON null. */
    VmafContext *vmaf;
    VmafConfiguration cfg = {0};
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    /* Single feature, single frame, value = NaN. */
    err = seed(vmaf->feature_collector, "feat_nan", NAN, 0);
    mu_assert("seed feat_nan failed", !err);
    /* Aggregate with +Inf and a normal sibling -> exercises both inf branch
     * and the trailing-comma logic between two aggregates. */
    err |= vmaf_feature_collector_set_aggregate(vmaf->feature_collector, "agg_inf", INFINITY);
    err |= vmaf_feature_collector_set_aggregate(vmaf->feature_collector, "agg_ok", 3.14);
    mu_assert("set_aggregate failed", !err);

    FILE *f = tmpfile();
    mu_assert("tmpfile failed", f);

    /* fps=NaN -> top-level "fps": null branch in vmaf_write_output_json. */
    err = vmaf_write_output_json(vmaf, vmaf->feature_collector, f, /*subsample=*/0,
                                 /*fps=*/NAN, /*pic_cnt=*/1, NULL);
    mu_assert("json: writer returned non-zero", !err);

    char *out = slurp(f);
    mu_assert("slurp failed", out);

    /* fps null branch (FP_NAN). */
    mu_assert("json: NaN fps must serialize as \"fps\": null", strstr(out, "\"fps\": null"));
    /* Frame metric NaN -> per-feature null. */
    mu_assert("json: NaN frame metric must serialize as null", strstr(out, "\"feat_nan\": null"));
    /* Aggregate +Inf -> null. */
    mu_assert("json: Inf aggregate must serialize as null", strstr(out, "\"agg_inf\": null"));
    /* Aggregate normal value still emitted. */
    mu_assert("json: agg_ok must still appear", strstr(out, "\"agg_ok\":"));

    teardown(vmaf, f, out);
    return NULL;
}

static char *test_json_empty_collector()
{
    /* Zero features, zero frames — exercises the "no frames" branch where
     * max_capacity returns 0 and the for-loop body never executes. The
     * writer must still emit valid JSON skeleton. */
    VmafContext *vmaf;
    VmafConfiguration cfg = {0};
    int err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    FILE *f = tmpfile();
    mu_assert("tmpfile failed", f);

    err = vmaf_write_output_json(vmaf, vmaf->feature_collector, f, /*subsample=*/0,
                                 /*fps=*/30.0, /*pic_cnt=*/0, NULL);
    mu_assert("json: writer returned non-zero on empty collector", !err);

    char *out = slurp(f);
    mu_assert("slurp failed", out);
    mu_assert("json: empty collector still emits skeleton open '{'", out[0] == '{');
    mu_assert("json: empty collector emits frames block", strstr(out, "\"frames\":"));
    mu_assert("json: empty collector emits pooled_metrics block",
              strstr(out, "\"pooled_metrics\":"));
    mu_assert("json: empty collector emits aggregate_metrics block",
              strstr(out, "\"aggregate_metrics\":"));

    teardown(vmaf, f, out);
    return NULL;
}

/* Read entire file at `path` into a malloc'd NUL-terminated buffer.
 * Caller frees.  Returns NULL on failure or when the file is larger than
 * 64 KiB (tests never produce files that large). */
static char *slurp_path(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return NULL;
    char *out = slurp(f);
    (void)fclose(f);
    return out;
}

static char *test_vmaf_version()
{
    /* vmaf_version() must return a non-NULL string that looks like a
     * semver-ish version (contains at least one ASCII digit). */
    const char *ver = vmaf_version();
    mu_assert("vmaf_version returned NULL", ver != NULL);
    /* Verify at least one digit present — a version string with no digit
     * would be clearly wrong. */
    bool has_digit = false;
    for (const char *p = ver; *p; p++) {
        if (*p >= '0' && *p <= '9') {
            has_digit = true;
            break;
        }
    }
    mu_assert("vmaf_version string contains no digit", has_digit);
    return NULL;
}

static char *test_write_output_json_path()
{
    /* vmaf_write_output() — public path-based dispatcher — must produce a
     * well-formed JSON file for VMAF_OUTPUT_FORMAT_JSON. */
    VmafContext *vmaf;
    int err = seed_normal(&vmaf);
    mu_assert("seed_normal failed", !err);

    char tmp[] = "/tmp/test_write_output_XXXXXX";
    int fd = mkstemp(tmp);
    mu_assert("mkstemp failed", fd >= 0);
    (void)close(fd);

    err = vmaf_write_output(vmaf, tmp, VMAF_OUTPUT_FORMAT_JSON);
    mu_assert("vmaf_write_output(JSON) returned non-zero", !err);

    char *out = slurp_path(tmp);
    (void)unlink(tmp);
    mu_assert("slurp_path failed after vmaf_write_output", out);
    mu_assert("write_output json: missing '{' open brace", strstr(out, "{"));
    mu_assert("write_output json: missing frames block", strstr(out, "\"frames\":"));
    mu_assert("write_output json: missing feat_a score", strstr(out, "feat_a"));

    free(out);
    (void)vmaf_close(vmaf);
    return NULL;
}

static char *test_write_output_with_format_custom()
{
    /* vmaf_write_output_with_format() must honour a caller-supplied printf
     * format string.  "%.3f" of 80.0 yields "80.000"; the default "%.17g"
     * would yield "80" or "80.000000000000000" — never "80.000" exactly. */
    VmafContext *vmaf;
    int err = seed_normal(&vmaf);
    mu_assert("seed_normal failed", !err);

    char tmp[] = "/tmp/test_write_output_fmt_XXXXXX";
    int fd = mkstemp(tmp);
    mu_assert("mkstemp failed", fd >= 0);
    (void)close(fd);

    err = vmaf_write_output_with_format(vmaf, tmp, VMAF_OUTPUT_FORMAT_JSON, "%.3f");
    mu_assert("vmaf_write_output_with_format(JSON,%.3f) returned non-zero", !err);

    char *out = slurp_path(tmp);
    (void)unlink(tmp);
    mu_assert("slurp_path failed after vmaf_write_output_with_format", out);
    /* "%.3f" of 80.0 → "80.000" (exactly three decimal places). */
    mu_assert("write_output_with_format: custom format '80.000' not found", strstr(out, "80.000"));
    /* The default 17-significant-digit form must NOT appear — that would
     * indicate the format string was ignored. */
    mu_assert("write_output_with_format: default format leaked despite custom",
              !strstr(out, "80.00000000000000"));

    free(out);
    (void)vmaf_close(vmaf);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_csv_basic);
    mu_run_test(test_csv_subsample_and_custom_format);
    mu_run_test(test_sub_basic);
    mu_run_test(test_xml_einval_guards);
    mu_run_test(test_xml_basic);
    mu_run_test(test_json_basic_and_format);
    mu_run_test(test_json_nan_and_inf);
    mu_run_test(test_json_empty_collector);
    mu_run_test(test_vmaf_version);
    mu_run_test(test_write_output_json_path);
    mu_run_test(test_write_output_with_format_custom);
    return NULL;
}
