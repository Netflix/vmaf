#include <errno.h>
#include <math.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"

#include "vif.h"
#include "vif_options.h"
#include "picture_copy.h"

typedef struct VifState {
    size_t float_stride;
    float *ref;
    float *dist;
} VifState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    VifState *s = fex->priv;
    s->float_stride = sizeof(float) * w;
    s->ref = malloc(s->float_stride * h);
    if (!s->ref) goto fail;
    s->dist = malloc(s->float_stride * h);
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
    VifState *s = fex->priv;
    int err = 0;

    picture_copy(s->ref, ref_pic);
    picture_copy(s->dist, dist_pic);

    double score, score_num, score_den;
    double scores[8];
    err = compute_vif(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                      s->float_stride, s->float_stride,
                      &score, &score_num, &score_den, scores);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector, "float_vif",
                                        score, index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector, "float_vif_num",
                                        score_num, index);
    if (err) return err;
    err = vmaf_feature_collector_append(feature_collector, "float_vif_den",
                                        score_den, index);
    if (err) return err;

    const char *feature_names[8] = {
        "float_vif_num_scale_0", "float_vif_den_scale_0",
        "float_vif_num_scale_1", "float_vif_den_scale_1",
        "float_vif_num_scale_2", "float_vif_den_scale_2",
        "float_vif_num_scale_3", "float_vif_den_scale_3",
    };

    for (unsigned i = 0; i < 8; i++) {
        err = vmaf_feature_collector_append(feature_collector, feature_names[i],
                                            scores[i], index);
        if (err) return err;
    }

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    VifState *s = fex->priv;
    if (s->ref) free(s->ref);
    if (s->dist) free(s->dist);
    return 0;
}

VmafFeatureExtractor vmaf_fex_float_vif = {
    .name = "float_vif",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(VifState),
};
