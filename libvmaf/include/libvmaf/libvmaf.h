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

#ifndef __VMAF_H__
#define __VMAF_H__

#include <stdint.h>
#include <stdio.h>

#include "libvmaf/model.h"
#include "libvmaf/picture.h"
#include "libvmaf/feature.h"

#ifdef __cplusplus
extern "C" {
#endif

enum VmafLogLevel {
    VMAF_LOG_LEVEL_NONE = 0,
    VMAF_LOG_LEVEL_ERROR,
    VMAF_LOG_LEVEL_WARNING,
    VMAF_LOG_LEVEL_INFO,
    VMAF_LOG_LEVEL_DEBUG,
};

enum VmafOutputFormat {
    VMAF_OUTPUT_FORMAT_NONE = 0,
    VMAF_OUTPUT_FORMAT_XML,
    VMAF_OUTPUT_FORMAT_JSON,
    VMAF_OUTPUT_FORMAT_CSV,
    VMAF_OUTPUT_FORMAT_SUB,
};

enum VmafPoolingMethod {
    VMAF_POOL_METHOD_UNKNOWN = 0,
    VMAF_POOL_METHOD_MIN,
    VMAF_POOL_METHOD_MAX,
    VMAF_POOL_METHOD_MEAN,
    VMAF_POOL_METHOD_HARMONIC_MEAN,
    VMAF_POOL_METHOD_NB
};

typedef struct VmafConfiguration {
    enum VmafLogLevel log_level;
    unsigned n_threads;
    unsigned n_subsample;
    uint64_t cpumask;
    uint64_t gpumask;
} VmafConfiguration;

typedef struct VmafContext VmafContext;

/**
 * Allocate and open a VMAF instance.
 *
 * @param vmaf The VMAF instance to open.
 *             To be used in further libvmaf api calls.
 *             $vmaf will be set to the allocated context.
 *             Context should be cleaned up with `vmaf_close()` when finished.
 *
 * @param cfg  Configuration parameters.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_init(VmafContext **vmaf, VmafConfiguration cfg);

/**
 * Register feature extractors required by a specific `VmafModel`.
 * This may be called multiple times using different models.
 * In this case, the registered feature extractors will form a set, and any
 * features required by multiple models will only be extracted once.
 *
 * @param vmaf  The VMAF context allocated with `vmaf_init()`.
 *
 * @param model Opaque model context.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_use_features_from_model(VmafContext *vmaf, VmafModel *model);

/**
 * Register feature extractors required by a specific `VmafModelCollection`
 * Like `vmaf_use_features_from_model()`, this function may be called
 * multiple times using different model collections.
 *
 * @param vmaf             The VMAF context allocated with `vmaf_init()`.
 *
 * @param model_collection Opaque model collection context.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_use_features_from_model_collection(VmafContext *vmaf,
                                            VmafModelCollection *model_collection);

/**
 * Register specific feature extractor.
 * Useful when a specific/additional feature is required, usually one which
 * is not already provided by a model via `vmaf_use_features_from_model()`.
 * This may be called multiple times. `VmafContext` will take ownership of the
 * `VmafFeatureDictionary` (`opts_dict`). Use `vmaf_feature_dictionary_free()`
 * only in the case of failure.
 *
 * @param vmaf         The VMAF context allocated with `vmaf_init()`.
 *
 * @param feature_name Name of feature.
 *
 * @param opts_dict    Feature extractor options set via
 *                     `vmaf_feature_dictionary_set()`. If no special options
 *                     are required this parameter can be set to NULL.
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_use_feature(VmafContext *vmaf, const char *feature_name,
                     VmafFeatureDictionary *opts_dict);

/**
 * Import an external feature score.
 * Useful when pre-computed feature scores are available.
 * Also useful in the case where there is no libvmaf feature extractor
 * implementation for a required feature.
 *
 * @param vmaf         The VMAF context allocated with `vmaf_init()`.
 *
 * @param feature_name Name of feature.
 *
 * @param value        Score.
 *
 * @param index        Picture index.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_import_feature_score(VmafContext *vmaf, const char *feature_name,
                              double value, unsigned index);

/**
 * Read a pair of pictures and queue them for eventual feature extraction.
 * This should be called after feature extractors are registered via
 * `vmaf_use_features_from_model()` and/or `vmaf_use_feature()`.
 * `VmafContext` will take ownership of both `VmafPicture`s (`ref` and `dist`)
 * and `vmaf_picture_unref()`.
 *
 * When you're done reading pictures call this function again with both `ref`
 * and `dist` set to NULL to flush all feature extractors.
 *
 * @param vmaf  The VMAF context allocated with `vmaf_init()`.
 *
 * @param ref   Reference picture.
 *
 * @param dist  Distorted picture.
 *
 * @param index Picture index.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_read_pictures(VmafContext *vmaf, VmafPicture *ref, VmafPicture *dist,
                       unsigned index);

/**
 * Predict VMAF score at specific index.
 *
 * @param vmaf   The VMAF context allocated with `vmaf_init()`.
 *
 * @param model  Opaque model context.
 *
 * @param index  Picture index.
 *
 * @param score  Predicted score.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_score_at_index(VmafContext *vmaf, VmafModel *model, double *score,
                        unsigned index);

/**
 * Predict VMAF score at specific index, using a model collection.
 *
 * @param vmaf              The VMAF context allocated with `vmaf_init()`.
 *
 * @param model_collection  Opaque model collection context.
 *
 * @param index             Picture index.
 *
 * @param score             Predicted score.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_score_at_index_model_collection(VmafContext *vmaf,
                                         VmafModelCollection *model_collection,
                                         VmafModelCollectionScore *score,
                                         unsigned index);

/**
 * Fetch feature score at specific index.
 *
 * @param vmaf          The VMAF context allocated with `vmaf_init()`.
 *
 * @param feature_name  Name of the feature to fetch.
 *
 * @param index         Picture index.
 *
 * @param score         Score.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_feature_score_at_index(VmafContext *vmaf, const char *feature_name,
                                double *score, unsigned index);

/**
 * Pooled VMAF score for a specific interval.
 *
 * @param vmaf         The VMAF context allocated with `vmaf_init()`.
 *
 * @param model        Opaque model context.
 *
 * @param pool_method  Temporal pooling method to use.
 *
 * @param score        Pooled score.
 *
 * @param index_low    Low picture index of pooling interval.
 *
 * @param index_high   High picture index of pooling interval.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_score_pooled(VmafContext *vmaf, VmafModel *model,
                      enum VmafPoolingMethod pool_method, double *score,
                      unsigned index_low, unsigned index_high);

/**
 * Pooled VMAF score for a specific interval, using a model collection.
 *
 * @param vmaf              The VMAF context allocated with `vmaf_init()`.
 *
 * @param model_collection  Opaque model collection context.
 *
 * @param pool_method       Temporal pooling method to use.
 *
 * @param score             Pooled score.
 *
 * @param index_low         Low picture index of pooling interval.
 *
 * @param index_high        High picture index of pooling interval.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_score_pooled_model_collection(VmafContext *vmaf,
                                       VmafModelCollection *model_collection,
                                       enum VmafPoolingMethod pool_method,
                                       VmafModelCollectionScore *score,
                                       unsigned index_low, unsigned index_high);

/**
 * Pooled feature score for a specific interval.
 *
 * @param vmaf          The VMAF context allocated with `vmaf_init()`.
 *
 * @param feature_name  Name of the feature to fetch.
 *
 * @param pool_method   Temporal pooling method to use.
 *
 * @param score         Pooled score.
 *
 * @param index_low     Low picture index of pooling interval.
 *
 * @param index_high    High picture index of pooling interval.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_feature_score_pooled(VmafContext *vmaf, const char *feature_name,
                              enum VmafPoolingMethod pool_method, double *score,
                              unsigned index_low, unsigned index_high);

/**
 * Close a VMAF instance and free all associated memory.
 *
 * @param vmaf The VMAF instance to close.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_close(VmafContext *vmaf);

/**
 * Write VMAF stats to an output file.
 *
 * @param vmaf         The VMAF context allocated with `vmaf_init()`.
 *
 * @param output_path  Output file path.
 *
 * @param fmt          Output file format.
 *                     See `enum VmafOutputFormat` for options.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_write_output(VmafContext *vmaf, const char *output_path,
                      enum VmafOutputFormat fmt);

/**
 * Get libvmaf version.
 */
const char *vmaf_version(void);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_H__ */
