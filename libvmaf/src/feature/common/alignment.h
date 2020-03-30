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

#ifndef ALIGNMENT_H_
#define ALIGNMENT_H_

/* Required for AVX. */
#define VMAF_ALIGNMENT 32

int vmaf_floorn(int n, int m);
int vmaf_ceiln(int n, int m);
//same as above functions but only used when m is in pow of 2
int vmaf_floorn_2pow(int n, int m);
int vmaf_ceiln_2pow(int n, int m);

#endif // ALIGNMENT_H_
