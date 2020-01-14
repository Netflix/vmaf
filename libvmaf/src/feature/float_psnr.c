#include <math.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"
#include "psnr.h"
#include "picture_copy.h"

static int init(VmafFeatureExtractor *fex)
{
    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    int err = 0;

    size_t stride = sizeof(float) * ref_pic->w[0];
    float *ref = malloc(ref_pic->h[0] * stride);
    float *dist = malloc(dist_pic->h[0] * stride);

    picture_copy(ref, ref_pic);
    picture_copy(dist, dist_pic);

    double score;
    err = compute_psnr(ref, dist, ref_pic->w[0], ref_pic->h[0], stride, stride,
                       &score, 255., 60.);

    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector, "float_psnr", score,
                                        index);
    if (err) return err;

    free(ref);
    free(dist);
    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    return 0;
}

VmafFeatureExtractor vmaf_fex_float_psnr = {
    .name = "float_psnr",
    .init = init,
    .extract = extract,
    .close = close,
};
