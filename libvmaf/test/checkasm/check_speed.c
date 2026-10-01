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

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "config.h"
#include "cpu.h"
#include "feature/speed.h"
#if ARCH_AARCH64
#include "feature/arm64/speed_neon.h"
#endif

#if ARCH_X86
#include "feature/x86/speed_avx2.h"
#if HAVE_AVX512
#include "feature/x86/speed_avx512.h"
#endif
#endif

typedef double (*compute_cov_kernel_fn)(const float *data_x,
                                         const float *data_y,
                                         size_t stride_px, size_t height,
                                         size_t width, double mean_x,
                                         double mean_y);

static compute_cov_kernel_fn get_compute_cov_kernel(unsigned cpu_flags)
{
    compute_cov_kernel_fn fn = compute_cov_kernel_scalar;
#if ARCH_X86
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX2)
        fn = compute_cov_kernel_avx2;
#if HAVE_AVX512
    if (cpu_flags & VMAF_X86_CPU_FLAG_AVX512)
        fn = compute_cov_kernel_avx512;
#endif
#elif ARCH_AARCH64
    if (cpu_flags & VMAF_ARM_CPU_FLAG_NEON)
        fn = compute_cov_kernel_neon;
#else
    (void) cpu_flags;
#endif
    return fn;
}

#define MAX_WIDTH  256
#define MAX_HEIGHT 64
#define MAX_STRIDE (MAX_WIDTH + 3)
#define BUFFER_SIZE (MAX_STRIDE * MAX_HEIGHT + 4)

static const struct { size_t w, h; } sizes[] = {
    { 1,   1  },
    { 2,   3  },
    { 3,   2  },
    { 4,   4  },
    { 5,   3  },
    { 6,   2  },
    { 7,   3  },
    { 8,   2  },
    { 9,   3  },
    { 15,  7  },
    { 16,  16 },
    { 17,  5  },
    { 37,  9  },
    { 255, 63 },
    { MAX_WIDTH, MAX_HEIGHT },
};

static void check_compute_cov_kernel(void)
{
    CHECKASM_ALIGN(float data_x[BUFFER_SIZE]);
    CHECKASM_ALIGN(float data_y[BUFFER_SIZE]);

    checkasm_declare(double, const float *, const float *, size_t, size_t,
                      size_t, double, double);

    if (checkasm_check_func(get_compute_cov_kernel(checkasm_get_cpu_flags()),
                             "compute_cov_kernel"))
    {
        for (unsigned pattern = 0; pattern < 3; pattern++) {
            for (size_t i = 0; i < BUFFER_SIZE; i++) {
                if (pattern == 0) {
                    data_x[i] = ((float) checkasm_rand_uint32() / (float) UINT32_MAX) * 255.f;
                    data_y[i] = ((float) checkasm_rand_uint32() / (float) UINT32_MAX) * 255.f;
                } else if (pattern == 1) {
                    data_x[i] = 127.5f;
                    data_y[i] = 130.25f;
                } else {
                    data_x[i] = 127.5f + (i % 2 ? 0.125f : -0.125f);
                    data_y[i] = 130.25f + (i % 3 ? 0.25f : -0.25f);
                }
            }

            for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
                const size_t w = sizes[i].w, h = sizes[i].h;
                const double mean_x = 127.5, mean_y = 130.25;

                for (unsigned layout = 0; layout < 3; layout++) {
                    const size_t stride = layout == 0 ? w : w + 3;
                    const float *x = data_x + (layout == 2 ? 1 : 0);
                    const float *y = data_y + (layout == 2 ? 3 : 0);
                    const double ref = checkasm_call_ref(x, y, stride, h,
                                                        w, mean_x, mean_y);
                    const double new = checkasm_call_new(x, y, stride, h,
                                                        w, mean_x, mean_y);

                    const double tol = 1e-10 * (fabs(ref) + 1.0);
                    if (!isfinite(new) || fabs(ref - new) > tol) {
                        if (checkasm_fail())
                            fprintf(stderr, "%zux%zu, pattern %u, layout %u: "
                                    "expected %.17g, got %.17g\n", w, h,
                                    pattern, layout, ref, new);
                    }
                }
            }
        }

        for (size_t i = 0; i < BUFFER_SIZE; i++) {
            data_x[i] = ((float) checkasm_rand_uint32() / (float) UINT32_MAX) * 255.f;
            data_y[i] = ((float) checkasm_rand_uint32() / (float) UINT32_MAX) * 255.f;
        }

        checkasm_bench_new(data_x, data_y, MAX_WIDTH, MAX_HEIGHT, MAX_WIDTH,
                            127.5, 130.25);
    }
}

void checkasm_check_speed(void)
{
    check_compute_cov_kernel();
    checkasm_report("compute_cov_kernel");
}
