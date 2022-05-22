/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "libvmaf/libvmaf.h"
#include "libvmaf/feature.h"

#include "cpu.h"
#include "feature/feature_extractor.h"
#include "feature/feature_collector.h"
#include "fex_ctx_vector.h"
#include "log.h"
#include "model.h"
#include "output.h"
#include "picture.h"
#include "predict.h"
#include "thread_pool.h"
#include "vcs_version.h"

typedef struct VmafContext {
    VmafConfiguration cfg;
    VmafFeatureCollector *feature_collector;
    RegisteredFeatureExtractors registered_feature_extractors;
    VmafFeatureExtractorContextPool *fex_ctx_pool;
    VmafThreadPool *thread_pool;
    struct {
        unsigned w, h;
        enum VmafPixelFormat pix_fmt;
        unsigned bpc;
    } pic_params;
    unsigned pic_cnt;
    bool flushed;
} VmafContext;

int vmaf_init(VmafContext **vmaf, VmafConfiguration cfg)
{
    if (!vmaf) return -EINVAL;
    int err = 0;

    VmafContext *const v = *vmaf = malloc(sizeof(*v));
    if (!v) goto fail;
    memset(v, 0, sizeof(*v));
    v->cfg = cfg;

    vmaf_init_cpu();
    vmaf_set_cpu_flags_mask(~cfg.cpumask);

    vmaf_set_log_level(cfg.log_level);

    err = vmaf_feature_collector_init(&(v->feature_collector));
    if (err) goto free_v;
    err = feature_extractor_vector_init(&(v->registered_feature_extractors));
    if (err) goto free_feature_collector;

    if (v->cfg.n_threads > 0) {
        err = vmaf_thread_pool_create(&v->thread_pool, v->cfg.n_threads);
        if (err) goto free_feature_extractor_vector;
        err = vmaf_fex_ctx_pool_create(&v->fex_ctx_pool, v->cfg.n_threads);
        if (err) goto free_thread_pool;
    }

    return 0;

free_thread_pool:
    vmaf_thread_pool_destroy(v->thread_pool);
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

    vmaf_thread_pool_wait(vmaf->thread_pool);
    feature_extractor_vector_destroy(&(vmaf->registered_feature_extractors));
    vmaf_feature_collector_destroy(vmaf->feature_collector);
    vmaf_thread_pool_destroy(vmaf->thread_pool);
    vmaf_fex_ctx_pool_destroy(vmaf->fex_ctx_pool);
    free(vmaf);

    return 0;
}

int vmaf_import_feature_score(VmafContext *vmaf, const char *feature_name,
                              double value, unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;

    return vmaf_feature_collector_append(vmaf->feature_collector, feature_name,
                                         value, index);
}

int vmaf_use_feature(VmafContext *vmaf, const char *feature_name,
                     VmafFeatureDictionary *opts_dict)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;

    VmafDictionary *s = (VmafDictionary*) opts_dict;

    int err = 0;

    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name(feature_name);
    if (!fex) return -EINVAL;

    VmafDictionary *d = NULL;
    if (s) {
        err = vmaf_dictionary_copy(&s, &d);
        if (err) return err;
        err = vmaf_dictionary_free(&s);
        if (err) return err;
    }

    VmafFeatureExtractorContext *fex_ctx;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex, d);
    if (err) return err;

    RegisteredFeatureExtractors *rfe = &(vmaf->registered_feature_extractors);
    err = feature_extractor_vector_append(rfe, fex_ctx, 0);
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
        if (!fex) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "could not initialize feature extractor \"%s\"\n",
                     model->feature[i].name);
            return -EINVAL;
        }

        VmafFeatureExtractorContext *fex_ctx;
        VmafDictionary *d = NULL;
        if (model->feature[i].opts_dict) {
            err = vmaf_dictionary_copy(&model->feature[i].opts_dict, &d);
            if (err) return err;
        }
        err = vmaf_feature_extractor_context_create(&fex_ctx, fex, d);
        if (err) return err;
        err = feature_extractor_vector_append(rfe, fex_ctx, 0);
        if (err) {
            err |= vmaf_feature_extractor_context_destroy(fex_ctx);
            return err;
        }
    }
    return 0;
}

int vmaf_use_features_from_model_collection(VmafContext *vmaf,
                                            VmafModelCollection *model_collection)
{
    if (!vmaf) return -EINVAL;
    if (!model_collection) return -EINVAL;

    int err = 0;
    for (unsigned i = 0; i < model_collection->cnt; i++)
        err |= vmaf_use_features_from_model(vmaf, model_collection->model[i]);

    return err;
}

struct ThreadData {
    VmafFeatureExtractorContext *fex_ctx;
    VmafPicture ref, dist;
    unsigned index;
    VmafFeatureCollector *feature_collector;
    VmafFeatureExtractorContextPool *fex_ctx_pool;
    int err;
};

static void threaded_extract_func(void *e)
{
    struct ThreadData *f = e;

    f->err = vmaf_feature_extractor_context_extract(f->fex_ctx, &f->ref, NULL,
                                                    &f->dist, NULL, f->index,
                                                    f->feature_collector);
    f->err = vmaf_fex_ctx_pool_release(f->fex_ctx_pool, f->fex_ctx);
    vmaf_picture_unref(&f->ref);
    vmaf_picture_unref(&f->dist);
}

static int threaded_read_pictures(VmafContext *vmaf, VmafPicture *ref,
                                  VmafPicture *dist, unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!ref) return -EINVAL;
    if (!dist) return -EINVAL;

    int err = 0;

    for (unsigned i = 0; i < vmaf->registered_feature_extractors.cnt; i++) {
        VmafFeatureExtractor *fex =
            vmaf->registered_feature_extractors.fex_ctx[i]->fex;
        VmafDictionary *opts_dict =
            vmaf->registered_feature_extractors.fex_ctx[i]->opts_dict;

        if ((vmaf->cfg.n_subsample > 1) && (index % vmaf->cfg.n_subsample) &&
            !(fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL))
        {
            continue;
        }

        VmafFeatureExtractorContext *fex_ctx;
        err = vmaf_fex_ctx_pool_aquire(vmaf->fex_ctx_pool, fex, opts_dict,
                                       &fex_ctx);
        if (err) return err;

        VmafPicture pic_a, pic_b;
        vmaf_picture_ref(&pic_a, ref);
        vmaf_picture_ref(&pic_b, dist);

        struct ThreadData data = {
            .fex_ctx = fex_ctx,
            .ref = pic_a,
            .dist = pic_b,
            .index = index,
            .feature_collector = vmaf->feature_collector,
            .fex_ctx_pool = vmaf->fex_ctx_pool,
            .err = 0,
        };

        err = vmaf_thread_pool_enqueue(vmaf->thread_pool, threaded_extract_func,
                                       &data, sizeof(data));
        if (err) {
            vmaf_picture_unref(&pic_a);
            vmaf_picture_unref(&pic_b);
            return err;
        }
    }

    return vmaf_picture_unref(ref) | vmaf_picture_unref(dist);
}

static int validate_pic_params(VmafContext *vmaf, VmafPicture *ref,
                               VmafPicture *dist)
{
    if (!vmaf->pic_params.w) {
        vmaf->pic_params.w = ref->w[0];
        vmaf->pic_params.h = ref->h[0];
        vmaf->pic_params.pix_fmt = ref->pix_fmt;
        vmaf->pic_params.bpc = ref->bpc;
    }

    if ((ref->w[0] != dist->w[0]) || (ref->w[0] != vmaf->pic_params.w))
        return -EINVAL;
    if ((ref->h[0] != dist->h[0]) || (ref->h[0] != vmaf->pic_params.h))
        return -EINVAL;
    if ((ref->pix_fmt != dist->pix_fmt) ||
        (ref->pix_fmt != vmaf->pic_params.pix_fmt))
    {
        return -EINVAL;
    }
    if ((ref->bpc != dist->bpc) && (ref->bpc != vmaf->pic_params.bpc))
        return -EINVAL;

    return 0;
}

static int flush_context_threaded(VmafContext *vmaf)
{
    int err = 0;
    err |= vmaf_thread_pool_wait(vmaf->thread_pool);
    err |= vmaf_fex_ctx_pool_flush(vmaf->fex_ctx_pool, vmaf->feature_collector);

    if (!err) vmaf->flushed = true;
    return err;
}

static int flush_context(VmafContext *vmaf)
{
    int err = 0;

    if (vmaf->thread_pool)
        return flush_context_threaded(vmaf);

    RegisteredFeatureExtractors rfe = vmaf->registered_feature_extractors;
    for (unsigned i = 0; i < rfe.cnt; i++) {
        err |= vmaf_feature_extractor_context_flush(rfe.fex_ctx[i],
                                                    vmaf->feature_collector);
    }

    if (!err) vmaf->flushed = true;
    return err;
}


int vmaf_read_pictures(VmafContext *vmaf, VmafPicture *ref, VmafPicture *dist,
                       unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (vmaf->flushed) return -EINVAL;
    if (!ref != !dist) return -EINVAL;
    if (!ref && !dist) return flush_context(vmaf);

    int err = 0;

    vmaf->pic_cnt++;
    err = validate_pic_params(vmaf, ref, dist);
    if (err) return err;

    if (vmaf->thread_pool)
        return threaded_read_pictures(vmaf, ref, dist, index);

    for (unsigned i = 0; i < vmaf->registered_feature_extractors.cnt; i++) {
        VmafFeatureExtractorContext *fex_ctx =
            vmaf->registered_feature_extractors.fex_ctx[i];

        if ((vmaf->cfg.n_subsample > 1) && (index % vmaf->cfg.n_subsample) &&
            !(fex_ctx->fex->flags & VMAF_FEATURE_EXTRACTOR_TEMPORAL))
        {
            continue;
        }

        err = vmaf_feature_extractor_context_extract(fex_ctx, ref, NULL, dist,
                                                     NULL, index,
                                                     vmaf->feature_collector);
        if (err) return err;
    }

    err = vmaf_picture_unref(ref);
    if (err) return err;
    err = vmaf_picture_unref(dist);
    if (err) return err;

    return 0;
}

int vmaf_feature_score_at_index(VmafContext *vmaf, const char *feature_name,
                                double *score, unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;
    if (!score) return -EINVAL;

    return vmaf_feature_collector_get_score(vmaf->feature_collector,
                                            feature_name, score, index);
}


int vmaf_score_at_index(VmafContext *vmaf, VmafModel *model, double *score,
                        unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!model) return -EINVAL;
    if (!score) return -EINVAL;

    int err =
        vmaf_feature_collector_get_score(vmaf->feature_collector, model->name,
                                         score, index);
    if (err) {
        err = vmaf_predict_score_at_index(model, vmaf->feature_collector, index,
                                          score, true, 0);
    }

    return err;
}

int vmaf_score_at_index_model_collection(VmafContext *vmaf,
                                         VmafModelCollection *model_collection,
                                         VmafModelCollectionScore *score,
                                         unsigned index)
{
    if (!vmaf) return -EINVAL;
    if (!model_collection) return -EINVAL;
    if (!score) return -EINVAL;

    return vmaf_predict_score_at_index_model_collection(model_collection,
                                                        vmaf->feature_collector,
                                                        index, score);
}

int vmaf_feature_score_pooled(VmafContext *vmaf, const char *feature_name,
                              enum VmafPoolingMethod pool_method, double *score,
                              unsigned index_low, unsigned index_high)
{
    if (!vmaf) return -EINVAL;
    if (!feature_name) return -EINVAL;
    if (index_low > index_high) return -EINVAL;
    if (!pool_method) return -EINVAL;

    unsigned pic_cnt = 0;
    double min = 0., max = 0., sum = 0., i_sum = 0.;
    for (unsigned i = index_low; i <= index_high; i++) {
        if ((vmaf->cfg.n_subsample > 1) && (i % vmaf->cfg.n_subsample))
            continue;
        pic_cnt++;
        double s;
        int err = vmaf_feature_score_at_index(vmaf, feature_name, &s, i);
        if (err) return err;
        sum += s;
        i_sum += 1. / (s + 1.);
        if ((i == index_low) || (s < min))
            min = s;
        if ((i == index_low) || (s > max))
            max = s;
    }

    switch (pool_method) {
    case VMAF_POOL_METHOD_MEAN:
        *score = sum / pic_cnt;
        break;
    case VMAF_POOL_METHOD_MIN:
        *score = min;
        break;
    case VMAF_POOL_METHOD_MAX:
        *score = max;
        break;
    case VMAF_POOL_METHOD_HARMONIC_MEAN:
        *score = pic_cnt / i_sum - 1.0;
        break;
    default:
        return -EINVAL;
    }

    return 0;
}

int vmaf_score_pooled(VmafContext *vmaf, VmafModel *model,
                      enum VmafPoolingMethod pool_method, double *score,
                      unsigned index_low, unsigned index_high)
{
    if (!vmaf) return -EINVAL;
    if (!model) return -EINVAL;
    if (!score) return -EINVAL;
    if (index_low > index_high) return -EINVAL;
    if (!pool_method) return -EINVAL;

    for (unsigned i = index_low; i <= index_high; i++) {
        if ((vmaf->cfg.n_subsample > 1) && (i % vmaf->cfg.n_subsample))
            continue;
        double vmaf_score;
        int err = vmaf_score_at_index(vmaf, model, &vmaf_score, i);
        if (err) return err;
    }

    return vmaf_feature_score_pooled(vmaf, model->name, pool_method, score,
                                     index_low, index_high);
}

int vmaf_score_pooled_model_collection(VmafContext *vmaf,
                                       VmafModelCollection *model_collection,
                                       enum VmafPoolingMethod pool_method,
                                       VmafModelCollectionScore *score,
                                       unsigned index_low, unsigned index_high)
{
    if (!vmaf) return -EINVAL;
    if (!model_collection) return -EINVAL;
    if (!score) return -EINVAL;
    if (index_low > index_high) return -EINVAL;
    if (!pool_method) return -EINVAL;

    int err = 0;
    for (unsigned i = index_low; i <= index_high; i++) {
        if ((vmaf->cfg.n_subsample > 1) && (i % vmaf->cfg.n_subsample))
            continue;
        VmafModelCollectionScore s;
        err = vmaf_score_at_index_model_collection(vmaf, model_collection, &s, i);
        if (err) return err;
    }

    score->type = VMAF_MODEL_COLLECTION_SCORE_BOOTSTRAP;

    //TODO: dedupe, vmaf_bootstrap_predict_score_at_index()
    const char *suffix_lo = "_ci_p95_lo";
    const char *suffix_hi = "_ci_p95_hi";
    const char *suffix_bagging = "_bagging";
    const char *suffix_stddev = "_stddev";
    const size_t name_sz =
        strlen(model_collection->name) + strlen(suffix_lo) + 1;
    char name[name_sz];
    memset(name, 0, name_sz);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_bagging);
    err |= vmaf_feature_score_pooled(vmaf, name, pool_method,
                                     &score->bootstrap.bagging_score,
                                     index_low, index_high);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_stddev);
    err |= vmaf_feature_score_pooled(vmaf, name, pool_method,
                                     &score->bootstrap.stddev,
                                     index_low, index_high);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_lo);
    err |= vmaf_feature_score_pooled(vmaf, name, pool_method,
                                     &score->bootstrap.ci.p95.lo,
                                     index_low, index_high);

    snprintf(name, name_sz, "%s%s", model_collection->name, suffix_hi);
    err |= vmaf_feature_score_pooled(vmaf, name, pool_method,
                                     &score->bootstrap.ci.p95.hi,
                                     index_low, index_high);

    return err;
}

const char *vmaf_version(void)
{
    return VMAF_VERSION;
}

int vmaf_write_output(VmafContext *vmaf, const char *output_path,
                      enum VmafOutputFormat fmt)
{
    FILE *outfile = fopen(output_path, "w");
    if (!outfile) {
        fprintf(stderr, "could not open file: %s\n", output_path);
        return -EINVAL;
    }

    const double fps = vmaf->pic_cnt /
                ((double) (vmaf->feature_collector->timer.end -
                vmaf->feature_collector->timer.begin) / CLOCKS_PER_SEC);

    int ret = 0;
    switch (fmt) {
    case VMAF_OUTPUT_FORMAT_XML:
        ret = vmaf_write_output_xml(vmaf, vmaf->feature_collector, outfile,
                                    vmaf->cfg.n_subsample,
                                    vmaf->pic_params.w, vmaf->pic_params.h,
                                    fps);
        break;
    case VMAF_OUTPUT_FORMAT_JSON:
        ret = vmaf_write_output_json(vmaf, vmaf->feature_collector, outfile,
                                     vmaf->cfg.n_subsample, fps);
        break;
    case VMAF_OUTPUT_FORMAT_CSV:
        ret = vmaf_write_output_csv(vmaf->feature_collector, outfile,
                                    vmaf->cfg.n_subsample);
        break;
    case VMAF_OUTPUT_FORMAT_SUB:
        ret = vmaf_write_output_sub(vmaf->feature_collector, outfile,
                                    vmaf->cfg.n_subsample);
        break;
    default:
        ret = -EINVAL;
        break;
    }

    fclose(outfile);
    return ret;
}
