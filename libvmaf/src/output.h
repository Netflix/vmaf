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

#ifndef __VMAF_OUTPUT_H__
#define __VMAF_OUTPUT_H__

int vmaf_write_output_xml(VmafFeatureCollector *fc, FILE *outfile,
                          unsigned subsample, unsigned width, unsigned height,
                          double fps);

int vmaf_write_output_json(VmafFeatureCollector *fc, FILE *outfile,
                           unsigned subsample);

int vmaf_write_output_csv(VmafFeatureCollector *fc, FILE *outfile,
                           unsigned subsample);

int vmaf_write_output_sub(VmafFeatureCollector *fc, FILE *outfile,
                          unsigned subsample);

#endif /* __VMAF_OUTPUT_H__ */
