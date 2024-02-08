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

#ifndef __VMAF_SRC_CPU_H__
#define __VMAF_SRC_CPU_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "config.h"

#if ARCH_X86
#include "x86/cpu.h"
#elif ARCH_AARCH64
#include "arm/cpu.h"
#endif

void vmaf_init_cpu(void);
void vmaf_set_cpu_flags_mask(const unsigned mask);
unsigned vmaf_get_cpu_flags(void);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_SRC_CPU_H__ */
