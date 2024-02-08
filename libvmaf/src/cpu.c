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

#include "config.h"
#include "cpu.h"

static unsigned flags = 0;
static unsigned flags_mask = -1;

void vmaf_init_cpu(void)
{
#if ARCH_X86
    flags = vmaf_get_cpu_flags_x86();
#elif ARCH_AARCH64
    flags = vmaf_get_cpu_flags_arm();
#endif
}

void vmaf_set_cpu_flags_mask(const unsigned mask)
{
    flags_mask = mask;
}

unsigned vmaf_get_cpu_flags(void)
{
    return flags & flags_mask;
}
