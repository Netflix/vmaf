#include <math.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    return 0;
}

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
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
        unsigned peak = pow(2, ref_pic->bpc) - 1;
        double score = MIN(10 * log10(peak * peak / MAX(noise, eps)), psnr_max);

        const char *feature_name[3] = { "psnr_y", "psnr_cb", "psnr_cr" };
        err = vmaf_feature_collector_append(feature_collector, feature_name[i],
                                            score, index);
        if (err) return err;
    }

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    return 0;
}

VmafFeatureExtractor vmaf_fex_psnr = {
    .name = "psnr",
    .init = init,
    .extract = extract,
    .close = close,
};
