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

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"
#include "opt.h"

typedef struct PsnrState {
    bool enable_chroma;
    bool enable_mse;
    bool enable_apsnr;
    bool reduced_hbd_peak;
    uint32_t peak;
    double psnr_max[3];
    double min_sse;
    struct {
        uint64_t sse[3];
        uint64_t n_pixels[3];
    } apsnr;
} PsnrState;

static const VmafOption options[] = {
    {
        .name = "enable_chroma",
        .help = "enable calculation for chroma channels",
        .offset = offsetof(PsnrState, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    {
        .name = "enable_mse",
        .help = "enable MSE calculation",
        .offset = offsetof(PsnrState, enable_mse),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_apsnr",
        .help = "enable APSNR calculation",
        .offset = offsetof(PsnrState, enable_apsnr),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "reduced_hbd_peak",
        .help = "reduce hbd peak value to align with scaled 8-bit content",
        .offset = offsetof(PsnrState, reduced_hbd_peak),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "min_sse",
        .help = "constrain the minimum possible sse",
        .offset = offsetof(PsnrState, min_sse),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.0,
        .min = 0.0,
        .max = DBL_MAX,
    },
    { 0 }
};


static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    PsnrState *s = fex->priv;
    s->peak = s->reduced_hbd_peak ? 255 * 1 << (bpc - 8) : (1 << bpc) - 1;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        s->enable_chroma = false;

    for (unsigned i = 0; i < 3; i++) {
        if (s->min_sse != 0.0) {
            const int ss_hor = pix_fmt != VMAF_PIX_FMT_YUV444P;
            const int ss_ver = pix_fmt == VMAF_PIX_FMT_YUV420P;
            const double mse = s->min_sse / 
                (((i && ss_hor) ? w / 2 : w) * ((i && ss_ver) ? h / 2 : h));
            s->psnr_max[i] = ceil(10. * log10(s->peak * s->peak / mse));
        } else {
            s->psnr_max[i] = (6 * bpc) + 12;
        }
    }

    return 0;
}

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static char *mse_name[3] = { "mse_y", "mse_cb", "mse_cr" };
static char *psnr_name[3] = { "psnr_y", "psnr_cb", "psnr_cr" };

static int psnr(VmafPicture *ref_pic, VmafPicture *dist_pic,
                unsigned index, VmafFeatureCollector *feature_collector,
                PsnrState *s)
{
    const uint8_t peak = 255; // (1 << ref_pic->bpc) - 1;
    const unsigned n = s->enable_chroma ? 3 : 1;

    int err = 0;

    for (unsigned p = 0; p < n; p++) {
        uint8_t *ref = ref_pic->data[p];
        uint8_t *dis = dist_pic->data[p];

        uint64_t sse = 0;
        for (unsigned i = 0; i < ref_pic->h[p]; i++) {
            uint32_t sse_inner = 0;
            for (unsigned j = 0; j < ref_pic->w[p]; j++) {
                const int16_t e = ref[j] - dis[j];
                sse_inner += e * e;
            }
            sse += sse_inner;
            ref += ref_pic->stride[p];
            dis += dist_pic->stride[p];
        }

        if (s->enable_apsnr) {
            s->apsnr.sse[p] += sse;
            s->apsnr.n_pixels[p] += ref_pic->h[p] * ref_pic->w[p];
        }

        const double mse = ((double) sse) / (ref_pic->w[p] * ref_pic->h[p]);
        const double psnr =
            MIN(10. * log10(peak * peak / MAX(mse, 1e-16)), s->psnr_max[p]);

        err |= vmaf_feature_collector_append(feature_collector, psnr_name[p],
                                             psnr, index);
        if (s->enable_mse) {
            err |= vmaf_feature_collector_append(feature_collector, mse_name[p],
                                                 mse, index);
        }
    }

    return err;
}

static int psnr_hbd(VmafPicture *ref_pic, VmafPicture *dist_pic,
                    unsigned index, VmafFeatureCollector *feature_collector,
                    PsnrState *s)
{
    const unsigned n = s->enable_chroma ? 3 : 1;

    int err = 0;

    for (unsigned p = 0; p < n; p++) {
        uint16_t *ref = ref_pic->data[p];
        uint16_t *dis = dist_pic->data[p];

        uint64_t sse = 0;
        for (unsigned i = 0; i < ref_pic->h[p]; i++) {
            for (unsigned j = 0; j < ref_pic->w[p]; j++) {
                const uint32_t e = abs(ref[j] - dis[j]);
                sse += e * e;
            }
            ref += ref_pic->stride[p] / 2;
            dis += dist_pic->stride[p] / 2;
        }

        if (s->enable_apsnr) {
            s->apsnr.sse[p] += sse;
            s->apsnr.n_pixels[p] += ref_pic->h[p] * ref_pic->w[p];
        }

        const double mse = ((double) sse) / (ref_pic->w[p] * ref_pic->h[p]);
        const double psnr =
            MIN(10. * log10(s->peak * s->peak / MAX(mse, 1e-16)),
                s->psnr_max[p]);

        err |= vmaf_feature_collector_append(feature_collector, psnr_name[p],
                                             psnr, index);
        if (s->enable_mse) {
            err |= vmaf_feature_collector_append(feature_collector, mse_name[p],
                                                 mse, index);
        }
    }

    return err;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    PsnrState *s = fex->priv;

    (void) ref_pic_90;
    (void) dist_pic_90;

    switch(ref_pic->bpc) {
    case 8:
        return psnr(ref_pic, dist_pic, index, feature_collector, s);
    case 10:
    case 12:
    case 16:
        return psnr_hbd(ref_pic, dist_pic, index, feature_collector, s);
    default:
        return -EINVAL;
    }
}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    PsnrState *s = fex->priv;
    const char *apsnr_name[3] = { "apsnr_y", "apsnr_cb", "apsnr_cr" };

    int err = 0;
    if (s->enable_apsnr) {
        for (unsigned i = 0; i < 3; i++) {

            double apsnr = 10 * (log10(s->peak * s->peak) +
                                 log10(s->apsnr.n_pixels[i]) -
                                 log10(s->apsnr.sse[i]));

            double max_apsnr =
                ceil(10 * log10(s->peak * s->peak *
                                s->apsnr.n_pixels[i] *
                                2));

            err |=
                vmaf_feature_collector_set_aggregate(feature_collector,
                                                     apsnr_name[i],
                                                     MIN(apsnr, max_apsnr));
        }
    }

    return (err < 0) ? err : !err;
}

static const char *provided_features[] = {
    "psnr_y", "psnr_cb", "psnr_cr",
    NULL
};

VmafFeatureExtractor vmaf_fex_psnr = {
    .name = "psnr",
    .options = options,
    .init = init,
    .extract = extract,
    .flush = flush,
    .priv_size = sizeof(PsnrState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
};
