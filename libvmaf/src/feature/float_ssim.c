#include "feature_collector.h"
#include "feature_extractor.h"

#include "ssim.h"

static int init(VmafFeatureExtractor *fex)
{
    return 0;
}

void picture_copy(float *dst, VmafPicture *src)
{
    float *float_data = dst;
    uint8_t *data = src->data[0];

    for (unsigned i = 0; i < src->h[0]; i++) {
        for (unsigned j = 0; j < src->w[0]; j++) {
            float_data[j] = (float) data[j];
        }
        float_data += src->w[0];
        data += src->stride[0];
    }

    return;
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

    double score, l_score, c_score, s_score;
    err = compute_ssim(ref, dist, ref_pic->w[0], ref_pic->h[0], stride, stride,
                       &score, &l_score, &c_score, &s_score);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector, "float_ssim", score,
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

VmafFeatureExtractor vmaf_fex_float_ssim = {
    .name = "float_ssim",
    .init = init,
    .extract = extract,
    .close = close,
};
