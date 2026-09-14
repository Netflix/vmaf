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

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "config.h"
#include "cpu.h"
#include "feature/integer_motion.h"

#if ARCH_X86
#include "feature/x86/motion_avx2.h"
#if HAVE_AVX512
#include "feature/x86/motion_avx512.h"
#endif
#elif ARCH_AARCH64
#include "feature/arm64/motion_neon.h"
#endif

typedef uint64_t (*motion_pipeline_fn)(const uint8_t *prev, ptrdiff_t prev_stride,
                                        const uint8_t *cur, ptrdiff_t cur_stride,
                                        int32_t *y_row, unsigned w, unsigned h,
                                        unsigned bpc);

static motion_pipeline_fn get_pipeline_8(unsigned cpu_flags)
{
    motion_pipeline_fn fn = motion_score_pipeline_8;
#if ARCH_X86
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = motion_score_pipeline_8_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = motion_score_pipeline_8_avx512;
#endif
#elif ARCH_AARCH64
    if (cpu_flags & VMAF_ARM_CPU_FLAG_NEON)
        fn = motion_score_pipeline_8_neon;
#endif
    return fn;
}

static motion_pipeline_fn get_pipeline_16(unsigned cpu_flags)
{
#if ARCH_X86
    motion_pipeline_fn fn = motion_score_pipeline_16;
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = motion_score_pipeline_16_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = motion_score_pipeline_16_avx512;
#endif
    return fn;
#else
    (void) cpu_flags;
    return 0;
#endif
}

#define MAX_WIDTH  256
#define MAX_HEIGHT 32

static const struct { unsigned w, h; } sizes[] = {
    { 4,   4  },
    { 16,  16 },
    { 32,  18 },
    { 173, 31 },
    { MAX_WIDTH, MAX_HEIGHT },
};

static void check_motion_8(void)
{
    CHECKASM_ALIGN(uint8_t prev[MAX_WIDTH * MAX_HEIGHT]);
    CHECKASM_ALIGN(uint8_t cur[MAX_WIDTH * MAX_HEIGHT]);
    CHECKASM_ALIGN(int32_t y_row_c[MAX_WIDTH]);
    CHECKASM_ALIGN(int32_t y_row_a[MAX_WIDTH]);

    checkasm_declare(uint64_t, const uint8_t *, ptrdiff_t, const uint8_t *,
                      ptrdiff_t, int32_t *, unsigned, unsigned, unsigned);

    if (checkasm_check_func(get_pipeline_8(checkasm_get_cpu_flags()),
                             "motion_score_pipeline_8"))
    {
        INITIALIZE_BUF(prev);
        INITIALIZE_BUF(cur);

        for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
            const unsigned w = sizes[i].w, h = sizes[i].h;
            CLEAR_BUF(y_row_c);
            CLEAR_BUF(y_row_a);

            const uint64_t sad_c =
                checkasm_call_ref(prev, w, cur, w, y_row_c, w, h, 8);
            const uint64_t sad_a =
                checkasm_call_new(prev, w, cur, w, y_row_a, w, h, 8);

            if (sad_c != sad_a) {
                if (checkasm_fail())
                    fprintf(stderr, "%ux%u: expected %llu, got %llu\n", w, h,
                            (unsigned long long) sad_c,
                            (unsigned long long) sad_a);
            }
        }

        checkasm_bench_new(prev, MAX_WIDTH, cur, MAX_WIDTH, y_row_a,
                            MAX_WIDTH, MAX_HEIGHT, 8);
    }
}

static void check_motion_16(unsigned bpc)
{
    CHECKASM_ALIGN(uint16_t prev[MAX_WIDTH * MAX_HEIGHT]);
    CHECKASM_ALIGN(uint16_t cur[MAX_WIDTH * MAX_HEIGHT]);
    CHECKASM_ALIGN(int32_t y_row_c[MAX_WIDTH]);
    CHECKASM_ALIGN(int32_t y_row_a[MAX_WIDTH]);

    checkasm_declare(uint64_t, const uint8_t *, ptrdiff_t, const uint8_t *,
                      ptrdiff_t, int32_t *, unsigned, unsigned, unsigned);

    if (checkasm_check_func(get_pipeline_16(checkasm_get_cpu_flags()),
                             "motion_score_pipeline_16_%dbpc", bpc))
    {
        INITIALIZE_BUF(prev);
        INITIALIZE_BUF(cur);

        const uint16_t mask = (uint16_t) ((1 << bpc) - 1);
        for (size_t i = 0; i < MAX_WIDTH * MAX_HEIGHT; i++) {
            prev[i] &= mask;
            cur[i]  &= mask;
        }

        for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
            const unsigned w = sizes[i].w, h = sizes[i].h;
            const ptrdiff_t stride = (ptrdiff_t) w * sizeof(uint16_t);
            CLEAR_BUF(y_row_c);
            CLEAR_BUF(y_row_a);

            const uint64_t sad_c =
                checkasm_call_ref((const uint8_t *) prev, stride,
                                  (const uint8_t *) cur, stride,
                                  y_row_c, w, h, bpc);
            const uint64_t sad_a =
                checkasm_call_new((const uint8_t *) prev, stride,
                                  (const uint8_t *) cur, stride,
                                  y_row_a, w, h, bpc);

            if (sad_c != sad_a) {
                if (checkasm_fail())
                    fprintf(stderr, "%ux%u @%dbpc: expected %llu, got %llu\n",
                            w, h, bpc, (unsigned long long) sad_c,
                            (unsigned long long) sad_a);
            }
        }

        const ptrdiff_t stride = MAX_WIDTH * sizeof(uint16_t);
        checkasm_bench_new((const uint8_t *) prev, stride,
                            (const uint8_t *) cur, stride, y_row_a,
                            MAX_WIDTH, MAX_HEIGHT, bpc);
    }
}

void checkasm_check_motion(void)
{
    check_motion_8();
    checkasm_report("motion_score_pipeline_8");

    check_motion_16(10);
    check_motion_16(12);
    check_motion_16(16);
    checkasm_report("motion_score_pipeline_16");
}
