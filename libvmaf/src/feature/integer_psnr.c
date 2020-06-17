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
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"
#include "opt.h"

typedef struct PsnrState {
    bool enable_chroma;
} PsnrState;

static const VmafOption options[] = {
    {
        .name = "enable_chroma",
        .help = "enable PSNR calculation for chroma channels",
        .offset = offsetof(PsnrState, enable_chroma),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = true,
    },
    { NULL }
};

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static int psnr8(VmafPicture *ref_pic, VmafPicture *dist_pic,
                 unsigned index, VmafFeatureCollector *feature_collector,
                 bool enable_chroma)
{
    int err = 0;
    const unsigned n = enable_chroma ? 3 : 1;

    for (unsigned i = 0; i < n; i++) {
        uint8_t *ref = ref_pic->data[i];
        uint8_t *dist = dist_pic->data[i];

        double noise = 0.;
        for (unsigned j = 0; j < ref_pic->h[i]; j++) {
            for (unsigned k = 0; k < ref_pic->w[i]; k++) {
                double diff = ref[k] - dist[k];
                noise += diff * diff;
            }
            ref += ref_pic->stride[i];
            dist += dist_pic->stride[i];
        }
        noise /= (ref_pic->w[i] * ref_pic->h[i]);

        double eps = 1e-10;
        double psnr_max = 60.;
        double peak = 255.0;
        double score = MIN(10 * log10(peak * peak / MAX(noise, eps)), psnr_max);

        const char *feature_name[3] = { "psnr_y", "psnr_cb", "psnr_cr" };
        err = vmaf_feature_collector_append(feature_collector, feature_name[i],
                                            score, index);
        if (err) return err;
    }

    return 0;
}

static int psnr10(VmafPicture *ref_pic, VmafPicture *dist_pic,
                  unsigned index, VmafFeatureCollector *feature_collector,
                  bool enable_chroma)
{
    int err = 0;
    const unsigned n = enable_chroma ? 3 : 1;

    for (unsigned i = 0; i < n; i++) {
        uint16_t *ref = ref_pic->data[i];
        uint16_t *dist = dist_pic->data[i];

        double noise = 0.;
        for (unsigned j = 0; j < ref_pic->h[i]; j++) {
            for (unsigned k = 0; k < ref_pic->w[i]; k++) {
                double diff = (ref[k] / 4.0) - (dist[k] / 4.0);
                noise += diff * diff;
            }
            ref += (ref_pic->stride[i] / 2);
            dist += (dist_pic->stride[i] / 2);
        }
        noise /= (ref_pic->w[i] * ref_pic->h[i]);

        double eps = 1e-10;
        double psnr_max = 72.;
        double peak = 255.75;
        double score = MIN(10 * log10(peak * peak / MAX(noise, eps)), psnr_max);

        const char *feature_name[3] = { "psnr_y", "psnr_cb", "psnr_cr" };
        err = vmaf_feature_collector_append(feature_collector, feature_name[i],
                                            score, index);
        if (err) return err;
    }

    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    PsnrState *s = fex->priv;

    switch(ref_pic->bpc) {
    case 8:
        return psnr8(ref_pic, dist_pic, index, feature_collector,
                     s->enable_chroma);
    case 10:
        return psnr10(ref_pic, dist_pic, index, feature_collector,
                      s->enable_chroma);
    default:
        return -EINVAL;
    }
}

static const char *provided_features[] = {
    "psnr_y", "psnr_cb", "psnr_cr",
    NULL
};

VmafFeatureExtractor vmaf_fex_psnr = {
    .name = "psnr",
    .extract = extract,
    .options = options,
    .priv_size = sizeof(PsnrState),
    .provided_features = provided_features,
};
