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

#include <checkasm/checkasm.h>
#include <checkasm/test.h>

#include "config.h"
#include "cpu.h"

void checkasm_check_motion(void);
void checkasm_check_adm(void);
void checkasm_check_speed(void);
void checkasm_check_cambi(void);
void checkasm_check_vif(void);

static const CheckasmTest tests[] = {
    { "motion", checkasm_check_motion, NULL, NULL },
    { "adm",    checkasm_check_adm,    NULL, NULL },
    { "speed",  checkasm_check_speed,  NULL, NULL },
    { "cambi",  checkasm_check_cambi,  NULL, NULL },
    { "vif",    checkasm_check_vif,    NULL, NULL },
    {0}
};

static const CheckasmCpuInfo cpu_flags[] = {
#if ARCH_X86
    { "AVX2",   "avx2",   VMAF_X86_CPU_FLAG_AVX2,   0 },
#if HAVE_AVX512
    { "AVX512", "avx512", VMAF_X86_CPU_FLAG_AVX512, 0 },
#endif
#elif ARCH_AARCH64
    { "NEON",   "neon",   VMAF_ARM_CPU_FLAG_NEON,   0 },
#endif
    {0}
};

static void set_cpu_flags(CheckasmCpu cpu)
{
    vmaf_set_cpu_flags_mask((unsigned) cpu);
}

int main(int argc, const char *argv[])
{
    vmaf_init_cpu();

    CheckasmConfig config = {
        .tests         = tests,
        .cpu_flags     = cpu_flags,
        .cpu           = vmaf_get_cpu_flags(),
        .set_cpu_flags = set_cpu_flags,
    };

    return checkasm_main(&config, argc, argv);
}
