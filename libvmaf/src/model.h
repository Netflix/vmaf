#ifndef __VMAF_SRC_MODEL_H__
#define __VMAF_SRC_MODEL_H__

#include <stdbool.h>

enum VmafModelType {
    VMAF_MODEL_TYPE_UNKNOWN = 0,
    VMAF_MODEL_TYPE_SVM_NUSVR,
    VMAF_MODEL_BOOTSTRAP_SVM_NUSVR,
    VMAF_MODEL_RESIDUE_BOOTSTRAP_SVM_NUSVR,
};

enum VmafModelNormalizationType {
    VMAF_MODEL_NORMALIZATION_TYPE_UNKNOWN = 0,
    VMAF_MODEL_NORMALIZATION_TYPE_NONE,
    VMAF_MODEL_NORMALIZATION_TYPE_LINEAR_RESCALE,
};

typedef struct {
    char *name;
    double slope, intercept;
} VmafModelFeature;

typedef struct {
    char *path;
    enum VmafModelType type;
    VmafModelFeature *feature;
    unsigned n_features;
    struct {
        bool enabled;
        double min, max;
    } score_clip;
    enum VmafModelNormalizationType norm_type;
    struct {
        bool enabled;
        struct {
            bool enabled;
            double value;
        } p0, p1, p2;
        bool out_lte_in, out_gte_in;
    } score_transform;
    struct svm_model *svm;
} VmafModel;

#endif /* __VMAF_SRC_MODEL_H__ */
