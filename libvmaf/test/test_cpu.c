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

#include "cpu.h"
#include "config.h"

static char *test_cpu()
{
    unsigned flags = 0;

    flags = vmaf_get_cpu_flags();
    mu_assert("flags should be zero before vmaf_init_cpu()", !flags);
    vmaf_init_cpu();

#if ARCH_X86
    /*
    flags = vmaf_get_cpu_flags();
    mu_assert("flags should include AVX2", flags & VMAF_X86_CPU_FLAG_AVX2);
    unsigned mask = ~VMAF_X86_CPU_FLAG_AVX2;
    vmaf_set_cpu_flags_mask(mask);
    flags = vmaf_get_cpu_flags();
    mu_assert("flags should not include AVX2 after masking",
              !(flags & VMAF_X86_CPU_FLAG_AVX2));
    */
#endif

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_cpu);
    return NULL;
}
