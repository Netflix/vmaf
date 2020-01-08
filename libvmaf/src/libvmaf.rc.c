#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "feature/feature_extractor.h"
#include "feature/feature_collector.h"
#include "libvmaf/libvmaf.rc.h"
#include "model.h"
#include "picture.h"

typedef struct {
    VmafFeatureExtractor **fex;
    unsigned cnt, capacity;
} VmafFeatureExtractorVector;

typedef struct VmafContext {
    VmafConfiguration cfg;
    VmafFeatureCollector *feature_collector;
    VmafFeatureExtractorVector registered_feature_extractor;
} VmafContext;

static int feature_extractor_vector_init(VmafFeatureExtractorVector *fev)
{
    fev->cnt = 0;
    fev->capacity = 8;
    size_t sz = sizeof(*(fev->fex)) * fev->capacity;
    fev->fex = malloc(sz);
    if (!fev->fex) return -ENOMEM;
    memset(fev->fex, 0, sz);
    return 0;
}

static int feature_extractor_vector_append(VmafFeatureExtractorVector *fev,
                                           VmafFeatureExtractor *fex)
{
    if (!fev) return -EINVAL;
    if (!fex) return -EINVAL;

    for (unsigned i = 0; i < fev->cnt; i++) {
        if (fev->fex[i] == fex)
            return 0;
    }

    if (fev->cnt >= fev->capacity) {
        size_t capacity = fev->capacity * 2;
        VmafFeatureExtractor **fex =
            realloc(fev->fex, sizeof(*(fev->fex)) * capacity);
        if (!fex) return -ENOMEM;
        fev->fex = fex;
        fev->capacity = capacity;
        for (int i = fev->cnt; i < fev->capacity; i++)
            fev->fex[i] = NULL;
    }

    fev->fex[fev->cnt++] = fex;
    return 0;
}

static void feature_extractor_vector_destroy(VmafFeatureExtractorVector *fev)
{
    if (!fev) return;
    if (!fev->fex) return;
    free(fev->fex);
    return;
}

void vmaf_default_configuration(VmafConfiguration *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
}

int vmaf_init(VmafContext **vmaf, VmafConfiguration cfg)
{
    if (!vmaf) return -EINVAL;
    int err = 0;

    VmafContext *const v = *vmaf = malloc(sizeof(*v));
    if (!v) goto fail;
    memset(v, 0, sizeof(*v));
    v->cfg = cfg;

    err = vmaf_feature_collector_init(&(v->feature_collector));
    if (err) goto free_v;
    err = feature_extractor_vector_init(&(v->registered_feature_extractor));
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

    feature_extractor_vector_destroy(&(vmaf->registered_feature_extractor));
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

    VmafFeatureExtractor *fex = NULL;
    fex = get_feature_extractor_by_name(feature_name);
    if (!fex) return -EINVAL;

    VmafFeatureExtractorVector *fev = &(vmaf->registered_feature_extractor);
    return feature_extractor_vector_append(fev, fex);
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
    for (unsigned i = 0; i < vmaf->registered_feature_extractor.cnt; i++) {
        VmafFeatureExtractor *fex = vmaf->registered_feature_extractor.fex[i];
        err = fex->extract(fex, ref, dist, index, vmaf->feature_collector);
        if (err) return err;
    }

    err = vmaf_picture_unref(ref);
    if (err) return err;
    err = vmaf_picture_unref(dist);
    if (err) return err;

    return 0;
}

int vmaf_score_at_index(VmafContext *vmaf, VmafModel model, VmafScore *score,
                        unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!score) return -EINVAL;

    return 0;
}

int vmaf_score_pooled(VmafContext *vmaf, VmafModel model,
                      enum VmafPoolingMethod pool_method, VmafScore *score,
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

void vmaf_write_log(VmafContext *vmaf, FILE *logfile)
{
    return;
}
