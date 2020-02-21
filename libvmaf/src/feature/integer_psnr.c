#include <errno.h>
#include <math.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static int psnr8(VmafPicture *ref_pic, VmafPicture *dist_pic,
                 unsigned index, VmafFeatureCollector *feature_collector)
{
    int err = 0;

    for (unsigned i = 0; i < 3; i++) {
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
                  unsigned index, VmafFeatureCollector *feature_collector)
{
    int err = 0;

    for (unsigned i = 0; i < 3; i++) {
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
    switch(ref_pic->bpc) {
    case 8:
        return psnr8(ref_pic, dist_pic, index, feature_collector);
    case 10:
        return psnr10(ref_pic, dist_pic, index, feature_collector);
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
    .provided_features = provided_features,
};
