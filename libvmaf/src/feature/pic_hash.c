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
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"

#include "mem.h"
#include "psnr.h"
#include "picture_copy.h"

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *dist_pic,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MD5_CTX ref_md5, dist_md5;

    for (unsigned i = 0; i < 3; i++) {
        for (unsigned j = 0; j < ref_pic->h[0]; j++) {

        }
    }
    
    err = vmaf_feature_collector_append(feature_collector, "ref_md5",
                                        score, index);
}

static const char *provided_features[] = {
    "ref_md5", "dist_md5", NULL
};

VmafFeatureExtractor vmaf_fex_float_psnr = {
    .name = "picture_hash",
    .extract = extract,
    .provided_features = provided_features,
};
