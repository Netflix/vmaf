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
#include <stddef.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"
#include "opt.h"

#include "mem.h"
#include "ansnr.h"
#include "picture_copy.h"

typedef struct AnsnrState {
    bool enable_chroma;
    size_t float_stride;
    float *ref;
    float *dist;
    double peak;
    double psnr_max;
} AnsnrState;

static const VmafOption options[] = {
    {
        .name = "enable_chroma",
        .help = "enable ANSNR/ANPSNR calculation for chroma (Cb/Cr) channels",
        .offset = offsetof(AnsnrState, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {0}};

static const char *ansnr_name[3] = {"float_ansnr", "float_ansnr_cb", "float_ansnr_cr"};
static const char *anpsnr_name[3] = {"float_anpsnr", "float_anpsnr_cb", "float_anpsnr_cr"};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    AnsnrState *s = fex->priv;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        s->enable_chroma = false;

    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref)
        goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist)
        goto free_ref;

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

    return 0;

free_ref:
    free(s->ref);
fail:
    return -ENOMEM;
}

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    AnsnrState *s = fex->priv;
    int err = 0;

    (void)ref_pic_90;
    (void)dist_pic_90;

    const unsigned n_planes = s->enable_chroma ? 3u : 1u;

    for (unsigned p = 0; p < n_planes; p++) {
        picture_copy(s->ref, s->float_stride, ref_pic, -128, ref_pic->bpc, p);
        picture_copy(s->dist, s->float_stride, dist_pic, -128, dist_pic->bpc, p);

        double score;
        double score_psnr;
        err = compute_ansnr(s->ref, s->dist, ref_pic->w[p], ref_pic->h[p], s->float_stride,
                            s->float_stride, &score, &score_psnr, s->peak, s->psnr_max);
        if (err)
            return err;

        err = vmaf_feature_collector_append(feature_collector, ansnr_name[p], score, index);
        if (err)
            return err;
        err = vmaf_feature_collector_append(feature_collector, anpsnr_name[p], score_psnr, index);
        if (err)
            return err;
    }

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    AnsnrState *s = fex->priv;
    if (s->ref)
        aligned_free(s->ref);
    if (s->dist)
        aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {"float_ansnr", NULL};

// NOLINTNEXTLINE(misc-use-internal-linkage): extern symbol referenced by feature_extractor.c registry.
VmafFeatureExtractor vmaf_fex_float_ansnr = {
    .name = "float_ansnr",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(AnsnrState),
    .options = options,
    .provided_features = provided_features,
};
