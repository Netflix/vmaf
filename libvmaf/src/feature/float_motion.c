#include <errno.h>
#include <math.h>
#include <string.h>

#include "common/convolution.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "mem.h"
#include "motion.h"
#include "motion_tools.h"

#include "picture_copy.h"

typedef struct MotionState {
    size_t float_stride;
    float *ref;
    float *tmp;
    float *blur[3];
    unsigned index;
    double score;
} MotionState;

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    MotionState *s = fex->priv;

    s->float_stride = sizeof(float) * w;
    s->ref = aligned_malloc(s->float_stride * h, 32);
    s->tmp = aligned_malloc(s->float_stride * h, 32);
    s->blur[0] = aligned_malloc(s->float_stride * h, 32);
    s->blur[1] = aligned_malloc(s->float_stride * h, 32);
    s->blur[2] = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref || !s->tmp || !s->blur[0] || !s->blur[1] || !s->blur[2])
        goto fail;

    s->score = 0;
    return 0;

fail:
    if (s->ref) aligned_free(s->ref);
    if (s->blur[0]) aligned_free(s->blur[0]);
    if (s->blur[1]) aligned_free(s->blur[1]);
    if (s->blur[2]) aligned_free(s->blur[2]);
    if (s->tmp) aligned_free(s->tmp);
    return -ENOMEM;

}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    MotionState *s = fex->priv;
    int ret = vmaf_feature_collector_append(feature_collector,
                                            "'VMAF_feature_motion2_score'",
                                            s->score, s->index);
    return (ret < 0) ? ret : !ret;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MotionState *s = fex->priv;
    int err = 0;

    s->index = index;
    unsigned blur_idx_0 = (index + 0) % 3;
    unsigned blur_idx_1 = (index + 1) % 3;
    unsigned blur_idx_2 = (index + 2) % 3;

    picture_copy(s->ref, ref_pic, -128, ref_pic->bpc);
    convolution_f32_c_s(FILTER_5_s, 5, s->ref, s->blur[blur_idx_0], s->tmp,
                        ref_pic->w[0], ref_pic->h[0],
                        s->float_stride / sizeof(float),
                        s->float_stride / sizeof(float));

    if (index == 0)
        return vmaf_feature_collector_append(feature_collector,
                                             "'VMAF_feature_motion2_score'",
                                             0., index);

    double score;
    err = compute_motion(s->blur[blur_idx_2], s->blur[blur_idx_0],
                         ref_pic->w[0], ref_pic->h[0],
                         s->float_stride, s->float_stride, &score);
    if (err) return err;
    s->score = score;

    if (index == 1)
        return 0;
    
    double score2;
    err = compute_motion(s->blur[blur_idx_2], s->blur[blur_idx_1],
                         ref_pic->w[0], ref_pic->h[0],
                         s->float_stride, s->float_stride, &score2);
    if (err) return err;
    score2 = score2 < score ? score2 : score;
    err = vmaf_feature_collector_append(feature_collector,
                                        "'VMAF_feature_motion2_score'",
                                        score2, index - 1);
    if (err) return err;

    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    MotionState *s = fex->priv;

    if (s->ref) aligned_free(s->ref);
    if (s->blur[0]) aligned_free(s->blur[0]);
    if (s->blur[1]) aligned_free(s->blur[1]);
    if (s->blur[2]) aligned_free(s->blur[2]);
    if (s->tmp) aligned_free(s->tmp);
    return 0;
}

static const char *provided_features[] = {
    "'VMAF_feature_motion2_score'",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_motion = {
    .name = "float_motion",
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close,
    .priv_size = sizeof(MotionState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
};
