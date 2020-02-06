#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libvmaf/libvmaf.rc.h>

#include "feature/common/cpu.h"
#include "feature/feature_extractor.h"
#include "feature/feature_collector.h"
#include "fex_ctx_vector.h"
#include "model.h"
#include "output.h"
#include "picture.h"
#include "predict.h"
#include "thread_pool.h"

typedef struct VmafContext {
    VmafConfiguration cfg;
    VmafFeatureCollector *feature_collector;
    RegisteredFeatureExtractors registered_feature_extractors;
    VmafThreadPool *thread_pool;
} VmafContext;

enum vmaf_cpu cpu;
// ^ FIXME, this is a global in the old libvmaf
// A few wrapped floating point feature extractors rely on it being a global
// After we clean those up, We'll add this to the VmafContext

int vmaf_init(VmafContext **vmaf, VmafConfiguration cfg)
{
    if (!vmaf) return -EINVAL;
    int err = 0;

    cpu = cpu_autodetect(); //FIXME, see above

    VmafContext *const v = *vmaf = malloc(sizeof(*v));
    if (!v) goto fail;
    memset(v, 0, sizeof(*v));
    v->cfg = cfg;

    err = vmaf_feature_collector_init(&(v->feature_collector));
    if (err) goto free_v;
    err = feature_extractor_vector_init(&(v->registered_feature_extractors));
    if (err) goto free_feature_collector;

    if (v->cfg.n_threads > 1) {
        err = vmaf_thread_pool_create(&v->thread_pool, v->cfg.n_threads);
        if (err) goto free_feature_extractor_vector;
    }

    return 0;

free_feature_extractor_vector:
    feature_extractor_vector_destroy(&(v->registered_feature_extractors));
free_feature_collector:
    vmaf_feature_collector_destroy(v->feature_collector);
free_v:
    free(v);
fail:
    return -ENOMEM;
}

int vmaf_close(VmafContext *vmaf)
{
    if (!vmaf) return -EINVAL;

    feature_extractor_vector_destroy(&(vmaf->registered_feature_extractors));
    vmaf_feature_collector_destroy(vmaf->feature_collector);
    vmaf_thread_pool_destroy(vmaf->thread_pool);
    free(vmaf);

    return 0;
}

int vmaf_import_feature_score(VmafContext *vmaf, char *feature_name,
                              double value, unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;

    return vmaf_feature_collector_append(vmaf->feature_collector, feature_name,
                                         value, index);
}

int vmaf_use_feature(VmafContext *vmaf, const char *feature_name)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;

    int err = 0;

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name(feature_name);
    if (!fex) return -EINVAL;

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex);
    if (err) return err;

    RegisteredFeatureExtractors *rfe = &(vmaf->registered_feature_extractors);
    err = feature_extractor_vector_append(rfe, fex_ctx);
    if (err)
        err |= vmaf_feature_extractor_context_destroy(fex_ctx);

    return err;
}

int vmaf_use_features_from_model(VmafContext *vmaf, VmafModel *model)
{
    if (!vmaf) return -EINVAL;
    if (!model) return -EINVAL;

    int err = 0;

    RegisteredFeatureExtractors *rfe = &(vmaf->registered_feature_extractors);

    for (unsigned i = 0; i < model->n_features; i++) {
        VmafFeatureExtractor *fex =
            vmaf_get_feature_extractor_by_feature_name(model->feature[i].name);
        if (!fex) return -EINVAL;
        VmafFeatureExtractorContext *fex_ctx;
        err = vmaf_feature_extractor_context_create(&fex_ctx, fex);
        if (err) return err;
        err = feature_extractor_vector_append(rfe, fex_ctx);
        if (err) {
            err |= vmaf_feature_extractor_context_destroy(fex_ctx);
            return err;
        }

    }

    return 0;
}

int vmaf_read_pictures(VmafContext *vmaf, VmafPicture *ref, VmafPicture *dist,
                       unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!ref) return -EINVAL;
    if (!dist) return -EINVAL;

    int err = 0;

    //TODO: VmafThreadPool
    for (unsigned i = 0; i < vmaf->registered_feature_extractors.cnt; i++) {
        VmafFeatureExtractorContext *fex_ctx =
            vmaf->registered_feature_extractors.fex_ctx[i];

        if ((vmaf->cfg.n_subsample > 1) && (index % vmaf->cfg.n_subsample) &&
            !(fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL))
        {
            continue;
        }

        err = vmaf_feature_extractor_context_extract(fex_ctx, ref, dist, index,
                                                     vmaf->feature_collector);
        if (err) return err;
    }

    err = vmaf_picture_unref(ref);
    if (err) return err;
    err = vmaf_picture_unref(dist);
    if (err) return err;

    return 0;
}

int vmaf_score_at_index(VmafContext *vmaf, VmafModel *model, double *score,
                        unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!score) return -EINVAL;

    return vmaf_predict_score_at_index(model, vmaf->feature_collector, index,
                                       score);
}

int vmaf_score_pooled(VmafContext *vmaf, VmafModel *model,
                      enum VmafPoolingMethod pool_method, double *score,
                      unsigned index_low, unsigned index_high)
{
    if (!vmaf) return -EINVAL;
    if (!score) return -EINVAL;
    if (index_low >= index_high) return -EINVAL;

    RegisteredFeatureExtractors rfe = vmaf->registered_feature_extractors;
    for (unsigned i = 0; i < rfe.cnt; i++)
        vmaf_feature_extractor_context_close(rfe.fex_ctx[i]);

    double sum = 0.;
    for (unsigned i = index_low; i < index_high; i++) {
        if ((vmaf->cfg.n_subsample > 1) && (i % vmaf->cfg.n_subsample))
            continue;
        double vmaf_score;
        int err = vmaf_score_at_index(vmaf, model, &vmaf_score, i);
        if (err) return err;
        sum += vmaf_score;
    }
    *score = sum / (index_high - index_low);
    return 0;
}

const char *vmaf_version(void)
{
    return "RELEASE_CANDIDATE";
}

int vmaf_write_output(VmafContext *vmaf, FILE *outfile,
                      enum VmafOutputFormat fmt)
{
    RegisteredFeatureExtractors rfe = vmaf->registered_feature_extractors;
    for (unsigned i = 0; i < rfe.cnt; i++)
        vmaf_feature_extractor_context_close(rfe.fex_ctx[i]);

    switch (fmt) {
    case VMAF_OUTPUT_FORMAT_XML:
        return vmaf_write_output_xml(vmaf->feature_collector, outfile,
                                     vmaf->cfg.n_subsample);
    default:
        return 0;
    }
}
