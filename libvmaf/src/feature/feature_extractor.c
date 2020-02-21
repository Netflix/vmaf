#include <errno.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#include "feature_extractor.h"

extern VmafFeatureExtractor vmaf_fex_ssim;
extern VmafFeatureExtractor vmaf_fex_float_ssim;
extern VmafFeatureExtractor vmaf_fex_psnr;
extern VmafFeatureExtractor vmaf_fex_float_psnr;
extern VmafFeatureExtractor vmaf_fex_float_adm;
extern VmafFeatureExtractor vmaf_fex_float_vif;
extern VmafFeatureExtractor vmaf_fex_float_motion;
extern VmafFeatureExtractor vmaf_fex_float_ms_ssim;

static VmafFeatureExtractor *feature_extractor_list[] = {
    &vmaf_fex_ssim,
    &vmaf_fex_float_ssim,
    &vmaf_fex_psnr,
    &vmaf_fex_float_psnr,
    &vmaf_fex_float_adm,
    &vmaf_fex_float_vif,
    &vmaf_fex_float_motion,
    &vmaf_fex_float_ms_ssim,
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
        memset(priv, 0, f->fex->priv_size);
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
    if (!fex_ctx->fex->extract) return -EINVAL;

    if (fex_ctx->fex->init && !fex_ctx->is_initialized) {
        int err =
            vmaf_feature_extractor_context_init(fex_ctx, ref->pix_fmt, ref->bpc,
                                                ref->w[0], ref->h[0]);
        if (err) return err;
    }

    return fex_ctx->fex->extract(fex_ctx->fex, ref, dist, pic_index, vfc);
}

int vmaf_feature_extractor_context_flush(VmafFeatureExtractorContext *fex_ctx,
                                         VmafFeatureCollector *vfc)
{
    if (!fex_ctx) return -EINVAL;
    if (!fex_ctx->is_initialized) return -EINVAL;
    if (fex_ctx->is_closed) return 0;

    int err = 0;
    if (fex_ctx->fex->flush)
        while (!(err = fex_ctx->fex->flush(fex_ctx->fex, vfc)));
    return err < 0 ? err : 0;
}

int vmaf_feature_extractor_context_close(VmafFeatureExtractorContext *fex_ctx)
{
    if (!fex_ctx) return -EINVAL;
    if (!fex_ctx->is_initialized) return -EINVAL;
    if (fex_ctx->is_closed) return 0;

    int err = 0;
    if (fex_ctx->fex->close)
        err = fex_ctx->fex->close(fex_ctx->fex);
    fex_ctx->is_closed = true;
    return err;
}

int vmaf_feature_extractor_context_destroy(VmafFeatureExtractorContext *fex_ctx)
{
    if (!fex_ctx) return -EINVAL;

    if (fex_ctx->fex) {
        if (fex_ctx->fex->priv)
            free(fex_ctx->fex->priv);
        free(fex_ctx->fex);
    }
    free(fex_ctx);
    return 0;
}

int vmaf_fex_ctx_pool_create(VmafFeatureExtractorContextPool **pool,
                             unsigned n_threads)
{
    if (!pool) return -EINVAL;
    if (!n_threads) return -EINVAL;

    VmafFeatureExtractorContextPool *const p = *pool = malloc(sizeof(*p));
    if (!p) return -ENOMEM;
    memset(p, 0, sizeof(*p));

    p->length =
        sizeof(feature_extractor_list) / sizeof(feature_extractor_list[0]) - 1;
    p->fex_list = malloc(p->length * sizeof(*(p->fex_list)));
    if (!p->fex_list) goto free_p;

    for (unsigned i = 0; i < p->length; i++) {
        VmafFeatureExtractor *fex = feature_extractor_list[i];
        p->fex_list[i].fex = fex;
        p->fex_list[i].capacity =
            fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL ? 1 : n_threads;
        p->fex_list[i].in_use = 0;
        pthread_cond_init(&(p->fex_list[i].full), NULL);
        size_t ctx_array_sz =
            sizeof(p->fex_list[i].ctx_list[0]) * p->fex_list[i].capacity;
        p->fex_list[i].ctx_list = malloc(ctx_array_sz);
        memset(p->fex_list[i].ctx_list, 0, ctx_array_sz);
        if (!p->fex_list[i].ctx_list)
            goto free_ctx_list;
    }

    pthread_mutex_init(&(p->lock), NULL);
    return 0;

free_ctx_list:
    for (unsigned i = 0; i < p->length; i++) {
        if (p->fex_list[i].ctx_list)
            free(p->fex_list[i].ctx_list);
    }
    free(p->fex_list);
free_p:
    free(p);
    return -ENOMEM;
}

int vmaf_fex_ctx_pool_aquire(VmafFeatureExtractorContextPool *pool,
                             VmafFeatureExtractor *fex,
                             VmafFeatureExtractorContext **fex_ctx)
{
    if (!pool) return -EINVAL;
    if (!fex) return -EINVAL;
    if (!fex_ctx) return -EINVAL;

    pthread_mutex_lock(&(pool->lock));
    int err = 0;

    struct fex_list_entry *entry = NULL;
    for (unsigned i = 0; i < pool->length; i++) {
        if (!strcmp(fex->name, pool->fex_list[i].fex->name)) {
            entry = &pool->fex_list[i];
            break;
        }
    }
    if (!entry) {
        err = -EINVAL;
        goto unlock;
    }

    while (entry->capacity == entry->in_use)
        pthread_cond_wait(&(entry->full), &(pool->lock));

    for (unsigned i = 0; i < entry->capacity; i++) {
        VmafFeatureExtractorContext *f = entry->ctx_list[i].fex_ctx;
        if (!f) {
            err = vmaf_feature_extractor_context_create(&f, entry->fex);
            if (err) goto unlock;
        }
        if (!entry->ctx_list[i].in_use) {
            entry->ctx_list[i].fex_ctx = *fex_ctx = f;
            entry->ctx_list[i].in_use = true;
            break;
        }
    }
    entry->in_use++;

unlock:
    pthread_mutex_unlock(&(pool->lock));
    return err;
}

int vmaf_fex_ctx_pool_release(VmafFeatureExtractorContextPool *pool,
                              VmafFeatureExtractorContext *fex_ctx)
{
    if (!pool) return -EINVAL;
    if (!fex_ctx) return -EINVAL;

    pthread_mutex_lock(&(pool->lock));
    int err = 0;

    VmafFeatureExtractor *fex = fex_ctx->fex;
    struct fex_list_entry *entry = NULL;
    for (unsigned i = 0; i < pool->length; i++) {
        if (!strcmp(fex->name, pool->fex_list[i].fex->name)) {
            entry = &pool->fex_list[i];
            break;
        }
    }
    if (!entry) {
        err = -EINVAL;
        goto unlock;
    }

    for (unsigned i = 0; i < entry->capacity; i++) {
        if (fex_ctx == entry->ctx_list[i].fex_ctx) {
            entry->ctx_list[i].in_use = false;
            entry->in_use--;
            pthread_cond_signal(&(entry->full));
            goto unlock;
        }
    }
    err = -EINVAL;

unlock:
    pthread_mutex_unlock(&(pool->lock));
    return err;
}

int vmaf_fex_ctx_pool_flush(VmafFeatureExtractorContextPool *pool,
                            VmafFeatureCollector *feature_collector)
{
    if (!pool) return -EINVAL;
    if (!pool->fex_list) return -EINVAL;
    pthread_mutex_lock(&(pool->lock));

    for (unsigned i = 0; i < pool->length; i++) {
        VmafFeatureExtractor *fex = pool->fex_list[i].fex;
        if (!(fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL))
            continue;
        for (unsigned j = 0; j < pool->fex_list[i].capacity; j++) {
            VmafFeatureExtractorContext *fex_ctx =
                pool->fex_list[i].ctx_list[j].fex_ctx;
            if (!fex_ctx) continue;
            vmaf_feature_extractor_context_flush(fex_ctx, feature_collector);
        }
    }

    pthread_mutex_unlock(&(pool->lock));
    return 0;
}

int vmaf_fex_ctx_pool_destroy(VmafFeatureExtractorContextPool *pool)
{
    if (!pool) return -EINVAL;
    if (!pool->fex_list) goto free_pool;
    pthread_mutex_lock(&(pool->lock));

    for (unsigned i = 0; i < pool->length; i++) {
        if (!pool->fex_list[i].ctx_list) continue;
        for (unsigned j = 0; j < pool->fex_list[i].capacity; j++) {
            VmafFeatureExtractorContext *fex_ctx =
                pool->fex_list[i].ctx_list[j].fex_ctx;
            if (!fex_ctx) continue;
            vmaf_feature_extractor_context_close(fex_ctx);
            vmaf_feature_extractor_context_destroy(fex_ctx);
        }
        free(pool->fex_list[i].ctx_list);
    }
    free(pool->fex_list);

free_pool:
    free(pool);
    return 0;
}
