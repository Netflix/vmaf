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

#include <errno.h>
#include <math.h>
#include <string.h>

#include "cpu.h"
#include "feature_collector.h"
#include "feature_extractor.h"

#include "mem.h"
#include "picture_copy.h"

#if ARCH_X86
#include "x86/float_psnr_avx2.h"
#if HAVE_AVX512
#include "x86/float_psnr_avx512.h"
#endif
#endif

#if ARCH_AARCH64
#include "arm64/float_psnr_neon.h"
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double float_psnr_noise_line_c(const float *ref, const float *dis, int w)
{
    double accum = 0;
    for (int j = 0; j < w; j++) {
        float diff = ref[j] - dis[j];
        accum += (double)(diff * diff);
    }
    return accum;
}

typedef struct PsnrState {
    size_t float_stride;
    float *ref;
    float *dist;
    double peak;
    double psnr_max;
    double (*noise_line)(const float *, const float *, int);
} PsnrState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;

    PsnrState *s = fex->priv;
    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref) goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist) goto free_ref;

    if (bpc == 8) {
        s->peak = 255.0;
        s->psnr_max = 60.0;
    } else if (bpc == 10) {
        s->peak = 255.75;
        s->psnr_max = 72.0;
    } else if (bpc == 12) {
        s->peak = 255.9375;
        s->psnr_max = 84.0;
    } else if (bpc == 16) {
        s->peak = 255.99609375;
        s->psnr_max = 108.0;
    } else {
        goto fail;
    }

    s->noise_line = float_psnr_noise_line_c;

#if ARCH_X86
    {
        unsigned flags = vmaf_get_cpu_flags();
        if (flags & VMAF_X86_CPU_FLAG_AVX2)
            s->noise_line = float_psnr_noise_line_avx2;
#if HAVE_AVX512
        if (flags & VMAF_X86_CPU_FLAG_AVX512)
            s->noise_line = float_psnr_noise_line_avx512;
#endif
    }
#elif ARCH_AARCH64
    {
        unsigned flags = vmaf_get_cpu_flags();
        if (flags & VMAF_ARM_CPU_FLAG_NEON)
            s->noise_line = float_psnr_noise_line_neon;
    }
#endif

    return 0;

free_ref:
    free(s->ref);
fail:
    return -ENOMEM;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    PsnrState *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;

    picture_copy(s->ref, s->float_stride, ref_pic, 0, ref_pic->bpc);
    picture_copy(s->dist, s->float_stride, dist_pic, 0, dist_pic->bpc);

    int w = ref_pic->w[0];
    int h = ref_pic->h[0];
    int stride = s->float_stride / sizeof(float);

    double noise_ = 0;
    for (int i = 0; i < h; i++)
        noise_ += s->noise_line(s->ref + i * stride, s->dist + i * stride, w);
    noise_ /= (w * h);

    double eps = 1e-10;
    double score = MIN(10 * log10(s->peak * s->peak / MAX(noise_, eps)),
                       s->psnr_max);
    err = vmaf_feature_collector_append(feature_collector, "float_psnr",
                                        score, index);
    if (err) return err;
    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    PsnrState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "float_psnr",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_psnr = {
    .name = "float_psnr",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(PsnrState),
    .provided_features = provided_features,
};
