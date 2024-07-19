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

#ifndef __VMAF_SRC_PROPAGATE_METADATA_H__
#define __VMAF_SRC_PROPAGATE_METADATA_H__

#include "libvmaf/libvmaf.h"

typedef struct VmafCallbackItem {
    void (*callback)(void *, VmafMetadata *);
    void *data;
    struct VmafCallbackItem *next;
} VmafCallbackItem;

typedef struct  VmafCallbackList{
    VmafCallbackItem *head;
} VmafCallbackList;

int vmaf_metadata_init(VmafCallbackList **const metadata);

int vmaf_metadata_append(VmafCallbackList *metadata,
                         const VmafMetadataConfig *metadata_config);

int vmaf_metadata_destroy(VmafCallbackList *metadata);

#endif // !__VMAF_PROPAGATE_METADATA_H__
