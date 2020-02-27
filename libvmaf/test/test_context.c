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

#include "test.h"
#include "libvmaf/libvmaf.rc.h"

static char *test_context_init_and_close()
{
    int err = 0;
    VmafContext *vmaf;
    VmafConfiguration cfg;

    vmaf_default_configuration(&cfg);

    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);
    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_context_init_and_close);
    return NULL;
}
