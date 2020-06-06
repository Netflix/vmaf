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

int vmaf_write_output_xml(VmafFeatureCollector *fc, FILE *outfile,
                          unsigned subsample, unsigned width, unsigned height,
                          double fps)
{
    if (!fc) return -EINVAL;
    if (!outfile) return -EINVAL;

    fprintf(outfile, "<VMAF version=\"%s\">\n", vmaf_version());
    fprintf(outfile, "  <params qualityWidth=\"%d\" qualityHeight=\"%d\" />\n",
            width, height);
    fprintf(outfile, "  <fyi fps=\"%.2f\" />\n", fps);

    fprintf(outfile, "  <frames>\n");
    for (unsigned i = 0 ; i < max_capacity(fc); i++) {
        if ((subsample > 1) && (i % subsample))
            continue;

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

int vmaf_write_output_json(VmafFeatureCollector *fc, FILE *outfile,
                           unsigned subsample)
{
    fprintf(outfile, "{\n");
    fprintf(outfile, "  \"version\": \"%s\",\n", vmaf_version());
    fprintf(outfile, "  \"frames\": [");

    for (unsigned i = 0 ; i < max_capacity(fc); i++) {
        if ((subsample > 1) && (i % subsample))
            continue;

        unsigned cnt = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (fc->feature_vector[j]->score[i].written)
                cnt++;
        }
        if (!cnt) continue;
        fprintf(outfile, "%s", i > 0 ? ",\n" : "\n");

        fprintf(outfile, "    {\n");
        fprintf(outfile, "      \"frameNum\": %d,\n", i);
        fprintf(outfile, "      \"metrics\": {\n");

        unsigned cnt2 = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (!fc->feature_vector[j]->score[i].written)
                continue;
            cnt2++;
            fprintf(outfile, "        \"%s\": %.6f%s\n",
                vmaf_feature_name_alias(fc->feature_vector[j]->name),
                fc->feature_vector[j]->score[i].value,
                cnt2 < cnt ? "," : ""
            );
        }
        fprintf(outfile, "      }\n");
        fprintf(outfile, "    }");
    }
    fprintf(outfile, "\n  ]\n");
    fprintf(outfile, "}\n");
    return 0;
}

int vmaf_write_output_csv(VmafFeatureCollector *fc, FILE *outfile,
                           unsigned subsample)
{

    fprintf(outfile, "Frame,");
    for (unsigned i = 0; i < fc->cnt; i++) {
        fprintf(outfile, "%s,",
                vmaf_feature_name_alias(fc->feature_vector[i]->name));
    }
    fprintf(outfile, "\n");

    for (unsigned i = 0 ; i < max_capacity(fc); i++) {
        if ((subsample > 1) && (i % subsample))
            continue;

        unsigned cnt = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (fc->feature_vector[j]->score[i].written)
                cnt++;
        }
        if (!cnt) continue;

        fprintf(outfile, "%d,", i);
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (!fc->feature_vector[j]->score[i].written)
                continue;
            fprintf(outfile, "%.6f,", fc->feature_vector[j]->score[i].value);
        }
        fprintf(outfile, "\n");
    }

    return 0;
}

int vmaf_write_output_sub(VmafFeatureCollector *fc, FILE *outfile,
                          unsigned subsample)
{
    for (unsigned i = 0 ; i < max_capacity(fc); i++) {
        if ((subsample > 1) && (i % subsample))
            continue;

        unsigned cnt = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (fc->feature_vector[j]->score[i].written)
                cnt++;
        }
        if (!cnt) continue;

        fprintf(outfile, "{%d}{%d}frame: %d|", i, i + 1, i);
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (!fc->feature_vector[j]->score[i].written)
                continue;
            fprintf(outfile, "%s: %.6f|",
                    vmaf_feature_name_alias(fc->feature_vector[j]->name),
                    fc->feature_vector[j]->score[i].value);
        }
        fprintf(outfile, "\n");
    }

    return 0;
}
