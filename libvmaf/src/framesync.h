/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#ifndef __VMAF_FRAME_SYNC_H__
#define __VMAF_FRAME_SYNC_H__

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include "libvmaf/libvmaf.h"

typedef struct VmafFrameSyncContext VmafFrameSyncContext;

int vmaf_framesync_init(VmafFrameSyncContext **fs_ctx);

int vmaf_framesync_acquire_new_buf(VmafFrameSyncContext *fs_ctx, void **data,
                                   unsigned data_sz, unsigned index);

int vmaf_framesync_submit_filled_data(VmafFrameSyncContext *fs_ctx, void *data,
                                      unsigned index);

int vmaf_framesync_retrieve_filled_data(VmafFrameSyncContext *fs_ctx, void **data,
                                        unsigned index);

int vmaf_framesync_release_buf(VmafFrameSyncContext *fs_ctx, void *data,
                               unsigned index);

int vmaf_framesync_destroy(VmafFrameSyncContext *fs_ctx);

#endif /* __VMAF_FRAME_SYNC_H__ */
