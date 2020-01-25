#include <errno.h>
#include <math.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"

#include "adm.h"
#include "adm_options.h"
#include "picture_copy.h"

typedef struct AdmState {
    size_t float_stride;
    float *ref;
    float *dist;
} AdmState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    AdmState *s = fex->priv;
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
    AdmState *s = fex->priv;
    int err = 0;

    picture_copy(s->ref, ref_pic);
    picture_copy(s->dist, dist_pic);

    double score, score_num, score_den;
    double scores[8];
    err = compute_adm(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                      s->float_stride, s->float_stride, &score, &score_num,
                      &score_den, scores, ADM_BORDER_FACTOR);
    if (err) return err;

    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_adm2_score'",
                                        score, index);

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    AdmState *s = fex->priv;
    if (s->ref) free(s->ref);
    if (s->dist) free(s->dist);
    return 0;
}

static const char *provided_features[] = {
    "'VMAF_feature_adm2_score'",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_adm = {
    .name = "float_adm",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(AdmState),
    .provided_features = provided_features,
};
