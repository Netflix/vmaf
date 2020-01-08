#ifndef __VMAF_H__
#define __VMAF_H__

#include <stdio.h>

#include "libvmaf/model.h"
#include "libvmaf/picture.h"

enum VmafLogLevel {
    VMAF_LOG_LEVEL_NONE,
};

typedef struct {
    enum VmafLogLevel log_level;
    unsigned n_threads;
} VmafConfiguration;

enum VmafPoolingMethod {
    VMAF_POOL_METHOD_UNKNOWN = 0,
    VMAF_POOL_METHOD_MIN,
    VMAF_POOL_METHOD_MEAN,
    VMAF_POOL_METHOD_HARMONIC_MEAN,
};

typedef struct {
    double score;
} VmafScore;

typedef struct VmafContext VmafContext;

void vmaf_default_configuration(VmafConfiguration *cfg);
void vmaf_write_log(VmafContext *vmaf, FILE *log);

/**
 * Get libvmaf version.
 */
const char *vmaf_version(void);

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
 * Register all feature extractors required by a model.
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
 * Register specific feature extractor.
 * Useful when a specific/additional feature is required, usually one which
 * is not provided by a model using `vmaf_use_features_from_model()`.
 * This may be called multiple times.
 *
 * @param vmaf         The VMAF context allocated with `vmaf_init()`.
 *
 * @param feature_name Name of feature.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_use_feature(VmafContext *vmaf, const char *feature_name);

/**
 * Import an external feature score.
 * Useful when pre-computed feature scores are available.
 * In this case, computation is saved since features have already been
 * extracted previously. Also useful in the case where there is no libvmaf
 * feature extractor implementation for a required feature.
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
int vmaf_import_feature_score(VmafContext *vmaf, char *feature_name,
                              double value, unsigned index);

/**
 * Read a pair of pictures and queue them for eventual feature extraction.
 * This should always be called after feature extractors are registered via
 * `vmaf_use_features_from_model()` and/or `vmaf_use_feature()`.
 *
 * @param vmaf  The VMAF context allocated with `vmaf_init()`.
 *
 * @param ref   Reference picture.
 *
 * @param dist  Distorted picture.
 *
 * @param index  Picture index.
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
int vmaf_score_at_index(VmafContext *vmaf, VmafModel model, VmafScore *score,
                        unsigned index);

/**
 * Predict pooled VMAF score for a specific interval.
 *
 * @param vmaf         The VMAF context allocated with `vmaf_init()`.
 *
 * @param model        Opaque model context.
 *
 * @param pool_method  Temporal pooling method to use.
 *
 * @param score        Predicted score.
 *
 * @param index_low    Low picture index of pooling interval.
 *
 * @param index_high   High picture index of pooling interval.
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_score_pooled(VmafContext *vmaf, VmafModel model,
                      enum VmafPoolingMethod pool_method, VmafScore *score,
                      unsigned index_low, unsigned index_high);

/**
 * Close a VMAF instance and free all associated memory.
 *
 * @param vmaf The VMAF instance to close.
 */
int vmaf_close(VmafContext *vmaf);

#endif /* __VMAF_H__ */
