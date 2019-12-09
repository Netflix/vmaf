#ifndef __VMAF_PREDICT_H__
#define __VMAF_PREDICT_H__

#include "feature/feature_collector.h"
#include "model.h"

int vmaf_predict_score_at_index(VmafModel *model,
                                VmafFeatureCollector *feature_collector,
                                unsigned index, double *vmaf_score);

#endif /* __VMAF_PREDICT_H__ */
