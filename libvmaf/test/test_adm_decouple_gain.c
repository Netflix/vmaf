/**
 *
 *  Copyright 2026 Lusoris
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

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "cpu.h"
#include "feature/integer_adm.h"
#include "test.h"

#if ARCH_X86
#include "feature/x86/adm_avx2.h"
#if HAVE_AVX512
#include "feature/x86/adm_avx512.h"
#endif
#endif

#if ARCH_X86

typedef void (*decouple_fn)(AdmBuffer *buf, int w, int h, int stride, double adm_enhn_gain_limit,
                            int32_t *adm_div_lookup);

/* A sample of the three bands at one position, reference and distorted. */
/* Bound of a scale 0 dwt2 coefficient, (54822 * 27395 + 32768) >> 16: the range a decoded
 * picture reaches. The scale 1-3 limit is a generous bound for their larger coefficients. */
#define DWT2_COEFF_MAX 22916
#define INT16_RANGE 32768
#define SCALE123_MAX (1 << 22)

typedef struct {
    int32_t o[3];
    int32_t t[3];
} Sample;

static uint32_t lcg_state;

static uint32_t lcg(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state >> 8;
}

static int32_t random_in(int32_t magnitude) {
    return (int32_t)(lcg() % (2u * (uint32_t)magnitude)) - magnitude;
}

static int32_t clamp_to(int64_t v, int32_t limit) {
    if (v < -(int64_t)limit) return -limit;
    if (v >= limit) return limit - 1;
    return (int32_t)v;
}

/* One position of bands whose samples lie in [-limit, limit). Three samples
 * in four have the distorted sample equal to the reference scaled by a factor
 * between 1/2 and 8 (and one in eight up to 150), which passes the one degree
 * angle test and makes adm_decouple() apply the enhancement gain limit; the
 * rest are independent. With the fractional gain limits below, rst * gain has
 * a fractional part in about every second limited sample. */
static Sample make_sample(int32_t limit) {
    Sample s;
    const uint32_t kind = lcg();
    const int32_t shifted = limit >> (kind % 12u);
    const int32_t magnitude = shifted < 4 ? 4 : shifted;
    const int64_t eighths = ((kind >> 8) % 8u == 0u) ? 1200 : 4 + (int64_t)((kind >> 12) % 60u);
    const int independent = ((kind >> 20) % 4u == 0u);

    for (int b = 0; b < 3; b++) {
        s.o[b] = random_in(magnitude);
        s.t[b] =
            independent ? random_in(magnitude) : clamp_to(((int64_t)s.o[b] * eighths) / 8, limit);
    }
    /* The angle test reads h and v only; d may point the other way. */
    if ((kind >> 24) & 1u) s.t[2] = clamp_to(-(int64_t)s.t[2], limit);
    return s;
}

typedef struct {
    int w, h, stride;
    AdmBuffer buf;
} Frame;

static int frame_alloc(Frame *f, int w, int h) {
    f->w = (w + 1) / 2;
    f->h = (h + 1) / 2;
    if (adm_buffer_alloc(&f->buf, w, h)) return -1;
    f->stride = (int)(f->buf.ind_size_x >> 2);
    return 0;
}

/* Fills the scale 0 (int16) or scale 1-3 (int32) reference and distorted
 * bands with random samples; every fifth position is taken from `special`
 * when it is given. */
static void frame_fill(Frame *f, int scale0, int32_t limit, const Sample *special, int n_special) {
    for (int i = 0; i < f->h * f->stride; i++) {
        const Sample s =
            (special && i % 5 == 0) ? special[(i / 5) % n_special] : make_sample(limit);
        if (scale0) {
            f->buf.ref_dwt2.band_h[i] = (int16_t)s.o[0];
            f->buf.ref_dwt2.band_v[i] = (int16_t)s.o[1];
            f->buf.ref_dwt2.band_d[i] = (int16_t)s.o[2];
            f->buf.dis_dwt2.band_h[i] = (int16_t)s.t[0];
            f->buf.dis_dwt2.band_v[i] = (int16_t)s.t[1];
            f->buf.dis_dwt2.band_d[i] = (int16_t)s.t[2];
        } else {
            f->buf.i4_ref_dwt2.band_h[i] = s.o[0];
            f->buf.i4_ref_dwt2.band_v[i] = s.o[1];
            f->buf.i4_ref_dwt2.band_d[i] = s.o[2];
            f->buf.i4_dis_dwt2.band_h[i] = s.t[0];
            f->buf.i4_dis_dwt2.band_v[i] = s.t[1];
            f->buf.i4_dis_dwt2.band_d[i] = s.t[2];
        }
    }
}

/* Samples of the restored and additive bands that differ. */
static size_t outputs_differ(const Frame *a, const Frame *b, int scale0) {
    size_t n = 0;
    const size_t count = (size_t)a->h * (size_t)a->stride;

    for (size_t i = 0; i < count; i++) {
        if (scale0) {
            n += a->buf.decouple_r.band_h[i] != b->buf.decouple_r.band_h[i];
            n += a->buf.decouple_r.band_v[i] != b->buf.decouple_r.band_v[i];
            n += a->buf.decouple_r.band_d[i] != b->buf.decouple_r.band_d[i];
            n += a->buf.decouple_a.band_h[i] != b->buf.decouple_a.band_h[i];
            n += a->buf.decouple_a.band_v[i] != b->buf.decouple_a.band_v[i];
            n += a->buf.decouple_a.band_d[i] != b->buf.decouple_a.band_d[i];
        } else {
            n += a->buf.i4_decouple_r.band_h[i] != b->buf.i4_decouple_r.band_h[i];
            n += a->buf.i4_decouple_r.band_v[i] != b->buf.i4_decouple_r.band_v[i];
            n += a->buf.i4_decouple_r.band_d[i] != b->buf.i4_decouple_r.band_d[i];
            n += a->buf.i4_decouple_a.band_h[i] != b->buf.i4_decouple_a.band_h[i];
            n += a->buf.i4_decouple_a.band_v[i] != b->buf.i4_decouple_a.band_v[i];
            n += a->buf.i4_decouple_a.band_d[i] != b->buf.i4_decouple_a.band_d[i];
        }
    }
    return n;
}

/* Runs the scalar kernel and `fn` on the same input and reports the number of
 * output samples that differ. -1 on allocation failure. */
static long compare(decouple_fn scalar, decouple_fn fn, int scale0, int w, int h, double gain,
                    int32_t limit, const Sample *special, int n_special) {
    Frame ref, simd;
    long bad = -1;

    if (frame_alloc(&ref, w, h)) return -1;
    if (frame_alloc(&simd, w, h)) {
        adm_buffer_free(&ref.buf);
        return -1;
    }

    lcg_state = 0x6a17u ^ (uint32_t)(w * 257 + h);
    frame_fill(&ref, scale0, limit, special, n_special);
    lcg_state = 0x6a17u ^ (uint32_t)(w * 257 + h);
    frame_fill(&simd, scale0, limit, special, n_special);

    scalar(&ref.buf, ref.w, ref.h, ref.stride, gain, div_lookup);
    fn(&simd.buf, simd.w, simd.h, simd.stride, gain, div_lookup);
    bad = (long)outputs_differ(&ref, &simd, scale0);

    adm_buffer_free(&ref.buf);
    adm_buffer_free(&simd.buf);
    return bad;
}

static int matches_scalar(decouple_fn scalar, decouple_fn fn, int scale0, const char *name,
                          int32_t limit, const Sample *special, int n_special) {
    static const double gains[] = {1.0, 1.2, 1.5, 100.0};
    static const int sizes[][2] = {{64, 48}, {97, 53}, {130, 72}};

    int ok = 1;

    for (unsigned g = 0; g < sizeof(gains) / sizeof(gains[0]); g++) {
        for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
            const long bad = compare(scalar, fn, scale0, sizes[s][0], sizes[s][1], gains[g], limit,
                                     special, n_special);
            if (bad != 0) {
                fprintf(stderr,
                        "%s, frame %dx%d, gain limit %g: %ld output samples differ from "
                        "the scalar kernel\n",
                        name, sizes[s][0], sizes[s][1], gains[g], bad);
                ok = 0;
            }
        }
    }
    return ok;
}

/* Positions with h = v = -32768 in both pictures: in the scale 0 kernels the
 * 16-bit multiply-add forms 2^31 for the squared magnitudes and the dot
 * product, which does not fit int32. Unreachable from a decoded picture
 * (scale 0 coefficients stay within 22916) but not from checkasm. */
static void bottom_samples(Sample *out, int n) {
    for (int i = 0; i < n; i++) {
        Sample s;
        memset(&s, 0, sizeof(s));
        s.o[0] = s.o[1] = s.t[0] = s.t[1] = -32768;
        s.o[2] = (i & 1) ? 1200 : -1200;
        s.t[2] = (i & 2) ? 1200 : -1200;
        out[i] = s;
    }
}

#endif

static char *test_adm_decouple_gain_limit_matches_scalar() {
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();
    int ok = 1;

    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        ok &= matches_scalar(adm_decouple, adm_decouple_avx2, 1, "adm_decouple_avx2",
                             DWT2_COEFF_MAX, NULL, 0);
        /* adm_decouple_s123_avx2 truncates the product in C and is not compared: it calls
         * get_best15_from32 on every lane, a negative shift for coefficients below 2^15. */
    }
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512) {
        ok &= matches_scalar(adm_decouple, adm_decouple_avx512, 1, "adm_decouple_avx512",
                             DWT2_COEFF_MAX, NULL, 0);
        ok &= matches_scalar(adm_decouple_s123, adm_decouple_s123_avx512, 0,
                             "adm_decouple_s123_avx512", SCALE123_MAX, NULL, 0);
    }
#endif
    mu_assert("a decouple kernel differs from the scalar kernel at a fractional gain limit", ok);
#endif
    return NULL;
}

static char *test_adm_decouple_angle_at_int16_minimum() {
#if ARCH_X86
    const unsigned flags = vmaf_get_cpu_flags();
    Sample special[64];
    int ok = 1;

    bottom_samples(special, 64);
    if (flags & VMAF_X86_CPU_FLAG_AVX2)
        ok &= matches_scalar(adm_decouple, adm_decouple_avx2, 1, "adm_decouple_avx2", INT16_RANGE,
                             special, 64);
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512)
        ok &= matches_scalar(adm_decouple, adm_decouple_avx512, 1, "adm_decouple_avx512",
                             INT16_RANGE, special, 64);
#endif
    mu_assert("a scale 0 decouple kernel differs from the scalar kernel at h = v = -32768", ok);
#endif
    return NULL;
}

char *run_tests() {
    vmaf_init_cpu();
#if ARCH_X86
    div_lookup_generator();
#endif
    mu_run_test(test_adm_decouple_gain_limit_matches_scalar);
    mu_run_test(test_adm_decouple_angle_at_int16_minimum);
    return NULL;
}
