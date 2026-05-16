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

#include <ctype.h>
#include <string.h>

#include "test.h"
#include "libvmaf/libvmaf.h"

/* Verify vmaf_version() returns a non-empty, printable, whitespace-free
 * string. The value comes from `git describe --tags --long --always`, so its
 * form varies: on a tagged release it is vMAJOR.MINOR.PATCH-K-gHASH; in a
 * dev build without a matching tag it falls back to a bare hex commit hash.
 * Both are valid; the assertion catches a NULL or empty return only.
 * Referenced in fuzz/meson.build but had no assertion-level coverage. */
static char *test_version_non_empty(void)
{
    const char *v = vmaf_version();
    mu_assert("vmaf_version() returned NULL", v != NULL);
    mu_assert("vmaf_version() returned empty string", v[0] != '\0');
    return NULL;
}

static char *test_version_printable(void)
{
    const char *v = vmaf_version();
    mu_assert("vmaf_version() returned NULL", v != NULL);

    /* Every character must be printable and non-whitespace: the string is
     * embedded verbatim into XML/JSON output (output.c) and must not
     * contain control characters or spaces. */
    for (const char *p = v; *p != '\0'; p++) {
        mu_assert("vmaf_version() contains non-printable character",
                  isprint((unsigned char)*p) && !isspace((unsigned char)*p));
    }
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_version_non_empty);
    mu_run_test(test_version_printable);
    return NULL;
}
