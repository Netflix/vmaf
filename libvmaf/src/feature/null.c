/**
 *
 *  Copyright 2016-2022 Netflix, Inc.
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

#include "feature_extractor.h"

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void) fex;
    (void) pix_fmt;
    (void) bpc;
    (void) w;
    (void) h;

    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    (void) fex;
    (void) ref_pic;
    (void) ref_pic_90;
    (void) dist_pic;
    (void) dist_pic_90;
    (void) index;
    (void) feature_collector;

    return 0;
}

VmafFeatureExtractor vmaf_fex_null = {
    .name = "null",
    .init = init,
    .extract = extract,
};
