#ifndef __VMAF_MODEL_H__
#define __VMAF_MODEL_H__

typedef struct VmafModel VmafModel;

int vmaf_model_load_from_path(VmafModel **model, const char *path);
void vmaf_model_destroy(VmafModel *model);

#endif /* __VMAF_MODEL_H__ */
