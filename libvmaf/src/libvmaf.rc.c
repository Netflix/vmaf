#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libvmaf/libvmaf.rc.h>

#include "feature/common/cpu.h"
#include "feature/feature_extractor.h"
#include "feature/feature_collector.h"
#include "model.h"
#include "output.h"
#include "picture.h"
#include "predict.h"

typedef struct {
    VmafFeatureExtractorContext **fex_ctx;
    unsigned cnt, capacity;
} RegisteredFeatureExtractors;

typedef struct VmafContext {
    VmafConfiguration cfg;
    VmafFeatureCollector *feature_collector;
    RegisteredFeatureExtractors registered_feature_extractors;
} VmafContext;

static int feature_extractor_vector_init(RegisteredFeatureExtractors *rfe)
{
    rfe->cnt = 0;
    rfe->capacity = 8;
    size_t sz = sizeof(*(rfe->fex_ctx)) * rfe->capacity;
    rfe->fex_ctx = malloc(sz);
    if (!rfe->fex_ctx) return -ENOMEM;
    memset(rfe->fex_ctx, 0, sz);
    return 0;
}

static int feature_extractor_vector_append(RegisteredFeatureExtractors *rfe,
                                           VmafFeatureExtractorContext *fex_ctx)
{
    if (!rfe) return -EINVAL;
    if (!fex_ctx) return -EINVAL;

    for (unsigned i = 0; i < rfe->cnt; i++) {
        if (rfe->fex_ctx[i] == fex_ctx)
            return 0;
    }

    if (rfe->cnt >= rfe->capacity) {
        size_t capacity = rfe->capacity * 2;
        VmafFeatureExtractorContext **fex_ctx =
            realloc(rfe->fex_ctx, sizeof(*(rfe->fex_ctx)) * capacity);
        if (!fex_ctx) return -ENOMEM;
        rfe->fex_ctx = fex_ctx;
        rfe->capacity = capacity;
        for (unsigned i = rfe->cnt; i < rfe->capacity; i++)
            rfe->fex_ctx[i] = NULL;
    }

    rfe->fex_ctx[rfe->cnt++] = fex_ctx;
    return 0;
}

static void feature_extractor_vector_destroy(RegisteredFeatureExtractors *rfe)
{
    if (!rfe) return;
    for (unsigned i = 0; i < rfe->cnt; i++) {
        vmaf_feature_extractor_context_close(rfe->fex_ctx[i]);
        vmaf_feature_extractor_context_destroy(rfe->fex_ctx[i]);
    }
    free(rfe->fex_ctx);
    return;
}

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
        fprintf(stderr, "set `--threads 1` for now.\n"
                        "multithreaded feature extraction is on the way.\n");
        return -EINVAL;
    }

    return 0;

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
    return feature_extractor_vector_append(rfe, fex_ctx);
}

int vmaf_use_features_from_model(VmafContext *vmaf, VmafModel *model)
{
    if (!vmaf) return -EINVAL;
    if (!model) return -EINVAL;

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

int vmaf_score_at_index(VmafContext *vmaf, VmafModel model, double *score,
                        unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!score) return -EINVAL;

    return vmaf_predict_score_at_index(&model, vmaf->feature_collector, index,
                                       score);
}

int vmaf_score_pooled(VmafContext *vmaf, VmafModel model,
                      enum VmafPoolingMethod pool_method, double *score,
                      unsigned index_low, unsigned index_high)
{
    if (!vmaf) return -EINVAL;
    if (!score) return -EINVAL;
    if (index_low >= index_high) return -EINVAL;

    return 0;
}

const char *vmaf_version(void)
{
    return "RELEASE_CANDIDATE";
}

int vmaf_write_output(VmafContext *vmaf, FILE *outfile,
                      enum VmafOutputFormat fmt)
{
    int err = 0;

    switch (fmt) {
    case VMAF_OUTPUT_FORMAT_NONE:
    case VMAF_OUTPUT_FORMAT_XML:
        err = vmaf_write_output_xml(vmaf->feature_collector, outfile);
        break;
    default:
        break;
    }

    return err;
}
