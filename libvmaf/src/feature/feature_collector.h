#ifndef __VMAF_FEATURE_COLLECTOR_H__
#define __VMAF_FEATURE_COLLECTOR_H__

typedef struct VmafFeatureCollector VmafFeatureCollector;

int vmaf_feature_collector_init(VmafFeatureCollector **const feature_collector);

int vmaf_feature_collector_append(VmafFeatureCollector *feature_collector,
                                  char *feature_name, double score,
                                  unsigned index);

int vmaf_feature_collector_get_score(VmafFeatureCollector *feature_collector,
                                     char *feature_name, double *score,
                                     unsigned index);

void vmaf_feature_collector_destroy(VmafFeatureCollector *feature_collector);

#endif /* __VMAF_FEATURE_COLLECTOR_H__ */
