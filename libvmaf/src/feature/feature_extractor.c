#include <errno.h>
#include <string.h>
#include <stdbool.h>

#include "feature_extractor.h"

extern VmafFeatureExtractor vmaf_fex_ssim;
extern VmafFeatureExtractor vmaf_fex_float_ssim;
extern VmafFeatureExtractor vmaf_fex_psnr;
extern VmafFeatureExtractor vmaf_fex_float_psnr;
extern VmafFeatureExtractor vmaf_fex_float_adm;
extern VmafFeatureExtractor vmaf_fex_float_vif;
extern VmafFeatureExtractor vmaf_fex_float_motion;

static VmafFeatureExtractor *feature_extractor_list[] = {
    &vmaf_fex_ssim,
    &vmaf_fex_float_ssim,
    &vmaf_fex_psnr,
    &vmaf_fex_float_psnr,
    &vmaf_fex_float_adm,
    &vmaf_fex_float_vif,
    &vmaf_fex_float_motion,
    NULL
};

VmafFeatureExtractor *vmaf_get_feature_extractor_by_name(char *name)
{
    if (!name) return NULL;

    VmafFeatureExtractor *fex = NULL;
    for (unsigned i = 0; (fex = feature_extractor_list[i]); i++) {
        if (!strcmp(name, fex->name))
           return fex;
    }
    return NULL;
}

VmafFeatureExtractor *vmaf_get_feature_extractor_by_feature_name(char *name)
{
    if (!name) return NULL;

    VmafFeatureExtractor *fex = NULL;
    for (unsigned i = 0; (fex = feature_extractor_list[i]); i++) {
        if (!fex->provided_features) continue;
        const char *fname = NULL;
        for (unsigned j = 0; (fname = fex->provided_features[j]); j++) {
            if (!strcmp(name, fname))
                return fex;
        }
    }
    return NULL;
}

int vmaf_feature_extractor_context_create(VmafFeatureExtractorContext **fex_ctx,
                                          VmafFeatureExtractor *fex)
{
    VmafFeatureExtractorContext *f = *fex_ctx = malloc(sizeof(*f));
    if (!f) return -ENOMEM;
    memset(f, 0, sizeof(*f));

    VmafFeatureExtractor *x = malloc(sizeof(*x));
    if (!x) goto free_f;
    memcpy(x, fex, sizeof(*x));

    f->fex = x;
    if (f->fex->priv_size) {
        void *priv = malloc(f->fex->priv_size);
        if (!priv) goto free_x;
        f->fex->priv = priv;
    }
    return 0;

free_x:
    free(x);
free_f:
    free(f);
    return -ENOMEM;
}

int vmaf_feature_extractor_context_init(VmafFeatureExtractorContext *fex_ctx,
                                        enum VmafPixelFormat pix_fmt,
                                        unsigned bpc, unsigned w, unsigned h)
{
    if (!fex_ctx) return -EINVAL;
    if (fex_ctx->is_initialized) return -EINVAL;
    if (!pix_fmt) return -EINVAL;

    int err = 0;
    if (!fex_ctx->is_initialized) {
        int err = fex_ctx->fex->init(fex_ctx->fex, pix_fmt, bpc, w, h);
        if (err) return err;
    }

    fex_ctx->is_initialized = true;
    return err;
}

int vmaf_feature_extractor_context_extract(VmafFeatureExtractorContext *fex_ctx,
                                           VmafPicture *ref, VmafPicture *dist,
                                           unsigned pic_index,
                                           VmafFeatureCollector *vfc)
{
    if (!fex_ctx) return -EINVAL;
    if (!ref) return -EINVAL;
    if (!dist) return -EINVAL;
    if (!vfc) return -EINVAL;
    if (!fex_ctx->fex->init) return -EINVAL;
    if (!fex_ctx->fex->extract) return -EINVAL;

    if (!fex_ctx->is_initialized) {
        int err =
            vmaf_feature_extractor_context_init(fex_ctx, ref->pix_fmt, ref->bpc,
                                                ref->w[0], ref->h[0]);
        if (err) return err;
    }

    return fex_ctx->fex->extract(fex_ctx->fex, ref, dist, pic_index, vfc);
}

int vmaf_feature_extractor_context_close(VmafFeatureExtractorContext *fex_ctx)
{
    if (!fex_ctx) return -EINVAL;
    if (!fex_ctx->is_initialized) return -EINVAL;
    if (fex_ctx->is_closed) return 0;

    int err = 0;
    if (fex_ctx->fex->close) {
        err = fex_ctx->fex->close(fex_ctx->fex);
    }
    fex_ctx->is_closed = true;
    return err;
}

int vmaf_feature_extractor_context_destroy(VmafFeatureExtractorContext *fex_ctx)
{
    if (!fex_ctx) return -EINVAL;

    if (fex_ctx->fex->priv_size)
        free(fex_ctx->fex->priv);
    free(fex_ctx->fex);
    free(fex_ctx);
    return 0;
}
