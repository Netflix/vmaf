#include <errno.h>

#include "feature_collector.h"
#include "feature_extractor.h"

#include "mem.h"
#include "ms_ssim.h"
#include "picture_copy.h"

typedef struct MsSsimState {
    size_t float_stride;
    float *ref;
    float *dist;
} MsSsimState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    MsSsimState *s = fex->priv;
    s->float_stride = sizeof(float) * w;
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref) goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist) goto free_ref;

    return 0;

free_ref:
    free(s->ref);
fail:
    return -ENOMEM;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MsSsimState *s = fex->priv;
    int err = 0;

    picture_copy(s->ref, ref_pic, 0, ref_pic->bpc);
    picture_copy(s->dist, dist_pic, 0, dist_pic->bpc);

    double score, l_scores[5], c_scores[5], s_scores[5];
    err = compute_ms_ssim(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                          s->float_stride, s->float_stride,
                          &score, l_scores, c_scores, s_scores);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector, "float_ms_ssim",
                                        score, index);
    if (err) return err;
    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    MsSsimState *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "float_ms_ssim",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_ms_ssim = {
    .name = "float_ms_ssim",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(MsSsimState),
    .provided_features = provided_features,
};
