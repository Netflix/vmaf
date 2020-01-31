#include <errno.h>
#include <stdio.h>

#include "feature/alias.h"
#include "feature/feature_collector.h"

#include <libvmaf/libvmaf.rc.h>

static unsigned max_capacity(VmafFeatureCollector *fc)
{
    unsigned capacity = 0;

    for (unsigned j = 0; j < fc->cnt; j++) {
        if (fc->feature_vector[j]->capacity > capacity)
            capacity = fc->feature_vector[j]->capacity;
    }

    return capacity;
}

int vmaf_write_output_xml(VmafFeatureCollector *fc, FILE *outfile)
{
    if (!fc) return -EINVAL;
    if (!outfile) return -EINVAL;

    fprintf(outfile, "<VMAF version=\"%s\">\n", vmaf_version());
    fprintf(outfile, "  <frames>\n");

    for (unsigned i = 0 ; i < max_capacity(fc); i++) {
        unsigned cnt = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (fc->feature_vector[j]->score[i].written)
                cnt++;
        }
        if (!cnt) continue;

        fprintf(outfile, "    <frame frameNum=\"%d\" ", i);
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (!fc->feature_vector[j]->score[i].written)
                continue;
            fprintf(outfile, "%s=\"%.6f\" ",
                vmaf_feature_name_alias(fc->feature_vector[j]->name),
                fc->feature_vector[j]->score[i].value
            );
        }
        fprintf(outfile, "/>\n");
    }

    fprintf(outfile, "  </frames>\n");
    fprintf(outfile, "</VMAF>\n");

    return 0;
}
