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

#include "libvmaf/propagate_metadata.h"

typedef struct VmafMetadataNode {
    void (*callback)(void *, const char *, double);
    void *data;
    size_t data_sz;
    struct VmafMetadataNode *next;
} VmafMetadataNode;

typedef struct  VmafMetadata{
    VmafMetadataNode *head;
} VmafMetadata;

int vmaf_metadata_init(VmafMetadata **const metadata);

int vmaf_metadata_append(VmafMetadata *metadata,
                                   const VmafMetadataConfig *metadata_config);

int vmaf_metadata_destroy(VmafMetadata *metadata);

#endif // !__VMAF_PROPAGATE_METADATA_H__
