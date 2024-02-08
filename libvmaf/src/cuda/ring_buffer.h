/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2021 NVIDIA Corporation.
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

#ifndef __VMAF_SRC_RING_BUFFER_H__
#define __VMAF_SRC_RING_BUFFER_H__

#include "picture.h"

typedef struct VmafRingBufferConfig {
    unsigned pic_cnt;
    int (*alloc_picture_callback)(VmafPicture *pic, void *cookie);
    int (*synchronize_picture_callback)(VmafPicture *pic, void *cookie);
    int (*free_picture_callback)(VmafPicture *pic, void *cookie);
    void *cookie;
} VmafRingBufferConfig;

typedef struct VmafRingBuffer VmafRingBuffer;

int vmaf_ring_buffer_init(VmafRingBuffer **ring_buffer, VmafRingBufferConfig cfg);

int vmaf_ring_buffer_close(VmafRingBuffer *ring_buffer);

// This function can have a latency as it calls the synchronize callback before fetching
int vmaf_ring_buffer_fetch_next_picture(VmafRingBuffer *ring_buffer, VmafPicture *pic);

#endif // __VMAF_SRC_RING_BUFFER_H__
