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

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"
#include "config.h"
#include "thread_locale.h"
#include "feature/feature_collector.h"
#include "output.h"
#include "read_json_model.h"

// Helper function to check if a locale is available
static int locale_available(const char *locale_name)
{
    char old_locale_buf[256];
    const char *old_locale = setlocale(LC_ALL, NULL);
    if (old_locale) {
        strncpy(old_locale_buf, old_locale, sizeof(old_locale_buf) - 1);
        old_locale_buf[sizeof(old_locale_buf) - 1] = '\0';
    } else {
        old_locale_buf[0] = '\0';
    }

    char *result = setlocale(LC_ALL, locale_name);
    int available = (result != NULL);

    if (old_locale_buf[0] != '\0') {
        setlocale(LC_ALL, old_locale_buf);
    }

    return available;
}

// Helper function to verify a string contains periods, not commas
static int contains_period_decimals(const char *str)
{
    // Check if we have numbers with periods (e.g., "123.456")
    // and NOT commas (e.g., "123,456")
    const char *p = str;
    int found_period = 0;

    while (*p) {
        if (*p >= '0' && *p <= '9') {
            // Found a digit, check next char
            if (*(p+1) == '.') {
                found_period = 1;
            } else if (*(p+1) == ',') {
                // Found comma decimal separator - FAIL
                return 0;
            }
        }
        p++;
    }

    return found_period;
}

static char *test_locale_abstraction_basic()
{
    // Test basic push/pop functionality
    if (!locale_available("es_ES.UTF-8") && !locale_available("es_ES.utf8")) {
        fprintf(stderr, "Skipping test: Spanish locale not available\n");
        return NULL;
    }

    const char *spanish = locale_available("es_ES.UTF-8") ? "es_ES.UTF-8" : "es_ES.utf8";
    setlocale(LC_ALL, spanish);

    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%.2f", 3.14);
    mu_assert("Spanish locale should use comma", strchr(buffer, ',') != NULL);

    VmafThreadLocaleState *state = vmaf_thread_locale_push_c();
    mu_assert("vmaf_thread_locale_push_c should not return NULL", state != NULL);

    snprintf(buffer, sizeof(buffer), "%.2f", 3.14);
    mu_assert("C locale should use period", strchr(buffer, '.') != NULL);
    mu_assert("C locale should not use comma", strchr(buffer, ',') == NULL);

    vmaf_thread_locale_pop(state);

    snprintf(buffer, sizeof(buffer), "%.2f", 3.14);
    mu_assert("Spanish locale should be restored", strchr(buffer, ',') != NULL);

    setlocale(LC_ALL, "C");

    return NULL;
}

static char *test_output_xml_with_comma_locale()
{
    if (!locale_available("fr_FR.UTF-8") && !locale_available("fr_FR.utf8")) {
        fprintf(stderr, "Skipping test: French locale not available\n");
        return NULL;
    }

    int err;
    VmafContext *vmaf;
    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 0,
        .n_subsample = 1,
    };

    err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("vmaf_feature_collector_init failed", !err);

    err = vmaf_feature_collector_append(fc, "test_feature", 12.345, 0);
    mu_assert("vmaf_feature_collector_append failed", !err);

    const char *french = locale_available("fr_FR.UTF-8") ? "fr_FR.UTF-8" : "fr_FR.utf8";
    setlocale(LC_ALL, french);

    FILE *tmpf = tmpfile();
    mu_assert("tmpfile creation failed", tmpf != NULL);

    err = vmaf_write_output_xml(vmaf, fc, tmpf, 1, 1920, 1080, 24.0, 1);
    mu_assert("vmaf_write_output_xml failed", !err);

    rewind(tmpf);
    char output[4096];
    size_t bytes_read = fread(output, 1, sizeof(output) - 1, tmpf);
    output[bytes_read] = '\0';
    fclose(tmpf);

    mu_assert("XML output should contain period decimals",
              contains_period_decimals(output));
    mu_assert("XML output should not contain 12,345",
              strstr(output, "12,345") == NULL);
    mu_assert("XML output should contain 12.3",
              strstr(output, "12.3") != NULL);

    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%.2f", 3.14);
    mu_assert("French locale should still be active", strchr(buffer, ',') != NULL);

    vmaf_feature_collector_destroy(fc);
    vmaf_close(vmaf);
    setlocale(LC_ALL, "C");

    return NULL;
}

static char *test_output_json_with_comma_locale()
{
    if (!locale_available("it_IT.UTF-8") && !locale_available("it_IT.utf8")) {
        fprintf(stderr, "Skipping test: Italian locale not available\n");
        return NULL;
    }

    int err;
    VmafContext *vmaf;
    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_NONE,
        .n_threads = 0,
        .n_subsample = 1,
    };

    err = vmaf_init(&vmaf, cfg);
    mu_assert("vmaf_init failed", !err);

    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("vmaf_feature_collector_init failed", !err);

    err = vmaf_feature_collector_append(fc, "test_feature", 98.765, 0);
    mu_assert("vmaf_feature_collector_append failed", !err);

    const char *italian = locale_available("it_IT.UTF-8") ? "it_IT.UTF-8" : "it_IT.utf8";
    setlocale(LC_ALL, italian);

    FILE *tmpf = tmpfile();
    mu_assert("tmpfile creation failed", tmpf != NULL);

    err = vmaf_write_output_json(vmaf, fc, tmpf, 1, 24.0, 1);
    mu_assert("vmaf_write_output_json failed", !err);

    rewind(tmpf);
    char output[4096];
    size_t bytes_read = fread(output, 1, sizeof(output) - 1, tmpf);
    output[bytes_read] = '\0';
    fclose(tmpf);

    mu_assert("JSON output should contain period decimals",
              contains_period_decimals(output));
    mu_assert("JSON output should not contain 98,765",
              strstr(output, "98,765") == NULL);

    // Verify Italian locale still active
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%.2f", 3.14);
    mu_assert("Italian locale should still be active", strchr(buffer, ',') != NULL);

    vmaf_feature_collector_destroy(fc);
    vmaf_close(vmaf);
    setlocale(LC_ALL, "C");

    return NULL;
}

static char *test_output_csv_with_comma_locale()
{
    if (!locale_available("es_ES.UTF-8") && !locale_available("es_ES.utf8")) {
        fprintf(stderr, "Skipping test: Spanish locale not available\n");
        return NULL;
    }

    int err;
    VmafFeatureCollector *fc;
    err = vmaf_feature_collector_init(&fc);
    mu_assert("vmaf_feature_collector_init failed", !err);

    err = vmaf_feature_collector_append(fc, "metric1", 45.678, 0);
    mu_assert("vmaf_feature_collector_append failed", !err);

    const char *spanish = locale_available("es_ES.UTF-8") ? "es_ES.UTF-8" : "es_ES.utf8";
    setlocale(LC_ALL, spanish);

    FILE *tmpf = tmpfile();
    mu_assert("tmpfile creation failed", tmpf != NULL);

    err = vmaf_write_output_csv(fc, tmpf, 1);
    mu_assert("vmaf_write_output_csv failed", !err);

    rewind(tmpf);
    char output[4096];
    size_t bytes_read = fread(output, 1, sizeof(output) - 1, tmpf);
    output[bytes_read] = '\0';
    fclose(tmpf);

    mu_assert("CSV output should contain period decimals",
              contains_period_decimals(output));

    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%.2f", 3.14);
    mu_assert("Spanish locale should still be active", strchr(buffer, ',') != NULL);

    vmaf_feature_collector_destroy(fc);
    setlocale(LC_ALL, "C");

    return NULL;
}

static char *test_model_parse_with_comma_locale()
{
    if (!locale_available("es_ES.UTF-8") && !locale_available("es_ES.utf8")) {
        fprintf(stderr, "Skipping test: Spanish locale not available\n");
        return NULL;
    }

    // Simple JSON model with floating point values
    const char *model_json = "{"
        "\"model_dict\":{"
        "\"model_type\":\"LIBSVMNUSVR\","
        "\"norm_type\":\"linear_rescale\","
        "\"slopes\":[1.5,2.5],"
        "\"intercepts\":[0.5,1.5],"
        "\"score_clip\":[0.0,100.0],"
        "\"feature_names\":[\"feature1\"]"
        "}}";

    const char *spanish = locale_available("es_ES.UTF-8") ? "es_ES.UTF-8" : "es_ES.utf8";
    setlocale(LC_ALL, spanish);

    // Verify Spanish locale active
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%.2f", 3.14);
    mu_assert("Spanish locale should be active", strchr(buffer, ',') != NULL);

    VmafModel *model = NULL;
    VmafModelConfig cfg = {
        .name = "test_model",
        .flags = VMAF_MODEL_FLAGS_DEFAULT,
    };

    int err = vmaf_read_json_model_from_buffer(&model, &cfg,
                                               model_json, strlen(model_json));
    mu_assert("vmaf_read_json_model_from_buffer should succeed", !err);
    mu_assert("model should not be NULL", model != NULL);

    // Verify values parsed correctly (with periods, not commas)
    mu_assert("slope should be 1.5", fabs(model->slope - 1.5) < 0.001);
    mu_assert("intercept should be 0.5", fabs(model->intercept - 0.5) < 0.001);

    // Verify Spanish locale still active
    snprintf(buffer, sizeof(buffer), "%.2f", 3.14);
    mu_assert("Spanish locale should still be active", strchr(buffer, ',') != NULL);

    vmaf_model_destroy(model);
    setlocale(LC_ALL, "C");

    return NULL;
}

#ifdef HAVE_USELOCALE
static char *test_uselocale_available()
{
    VmafThreadLocaleState *state = vmaf_thread_locale_push_c();
    mu_assert("HAVE_USELOCALE: push should succeed", state != NULL);
    vmaf_thread_locale_pop(state);
    return NULL;
}
#elif defined(_WIN32)
static char *test_windows_locale_handling()
{
    VmafThreadLocaleState *state = vmaf_thread_locale_push_c();
    mu_assert("Windows: push should succeed", state != NULL);
    vmaf_thread_locale_pop(state);
    return NULL;
}
#endif

char *run_tests()
{
    mu_run_test(test_locale_abstraction_basic);
    mu_run_test(test_output_xml_with_comma_locale);
    mu_run_test(test_output_json_with_comma_locale);
    mu_run_test(test_output_csv_with_comma_locale);
    mu_run_test(test_model_parse_with_comma_locale);

#ifdef HAVE_USELOCALE
    mu_run_test(test_uselocale_available);
#elif defined(_WIN32)
    mu_run_test(test_windows_locale_handling);
#endif

    return NULL;
}
