#ifndef __VMAF_MODEL_H__
#define __VMAF_MODEL_H__

typedef struct VmafModel VmafModel;

enum VmafModelFlags {
    VMAF_MODEL_FLAG_DISABLE_CLIP = (1 << 0),
    VMAF_MODEL_FLAG_ENABLE_TRANSFORM = (1 << 1),
    VMAF_MODEL_FLAG_ENABLE_CONFIDENCE_INTERVAL = (1 << 2),
};

#define VMAF_MODEL_FLAGS_DEFAULT VMAF_MODEL_FLAG_DISABLE_CLIP

typedef struct VmafModelConfig {
    enum VmafModelFlags flags;
    char *name;
    char *path;
} VmafModelConfig;

int vmaf_model_load_from_path(VmafModel **model, VmafModelConfig *cfg);
void vmaf_model_destroy(VmafModel *model);

#endif /* __VMAF_MODEL_H__ */
