#ifndef __VMAF_FEATURE_COLLECTOR_H__
#define __VMAF_FEATURE_COLLECTOR_H__

#include <pthread.h>
#include <stdbool.h>

typedef struct {
    char *name;
    struct {
        bool written;
        double value;
    } *score;
    unsigned capacity;
} FeatureVector;

typedef struct VmafFeatureCollector {
    FeatureVector **feature_vector;
    unsigned cnt, capacity;
    pthread_mutex_t lock;
} VmafFeatureCollector;

int vmaf_feature_collector_init(VmafFeatureCollector **const feature_collector);

int vmaf_feature_collector_append(VmafFeatureCollector *feature_collector,
                                  char *feature_name, double score,
                                  unsigned index);

int vmaf_feature_collector_get_score(VmafFeatureCollector *feature_collector,
                                     char *feature_name, double *score,
                                     unsigned index);

void vmaf_feature_collector_destroy(VmafFeatureCollector *feature_collector);

#endif /* __VMAF_FEATURE_COLLECTOR_H__ */
