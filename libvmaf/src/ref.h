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

#ifndef __VMAF_SRC_REF_H__
#define __VMAF_SRC_REF_H__

#include <stdatomic.h>

typedef struct VmafRef {
    atomic_int cnt;
} VmafRef;

int vmaf_ref_init(VmafRef **ref);
void vmaf_ref_fetch_increment(VmafRef *ref);
void vmaf_ref_fetch_decrement(VmafRef *ref);
long vmaf_ref_load(VmafRef *ref);
int vmaf_ref_close(VmafRef *ref);

#endif /* __VMAF_SRC_REF_H__ */
