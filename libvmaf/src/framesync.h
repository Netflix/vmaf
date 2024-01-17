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

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include "libvmaf/libvmaf.h"


// defined based on maximum temporal dependency across all feature extractors
// Motion score and STRRED features have dependency on previous frame. Dep is set to 1
#define MAX_TEMPORAL_DEP 1 
#define MAX_FRAME_SYNC_BUF_Q_LEN (MAX_VMAF_THREADS + MAX_TEMPORAL_DEP)

typedef enum
{
    BUFF_FREE = 0,
    BUFF_AQUIRED,
    BUFF_FILLED,
    BUFF_RETRIVED,
} BUFF_STATUS_T;

typedef struct VmafFrameSyncBuff {
    void          *frame_data;	
	unsigned      buf_status; //BUFF_STATUS_T
    signed long   index;	    
} VmafFrameSyncBuff;

typedef struct VmafFrameSyncContext {
  	VmafFrameSyncBuff buf_que[MAX_FRAME_SYNC_BUF_Q_LEN];
	pthread_mutex_t aquire_lock;
    pthread_cond_t  aquire;
	pthread_mutex_t retrive_lock;
	pthread_cond_t  retrive;
    unsigned        buf_cnt;    
} VmafFrameSyncContext; 

typedef struct VmafFrameSyncContext VmafFrameSyncContext;

int vmaf_framesync_init(VmafFrameSyncContext **fs_ctx);

int vmaf_framesync_aquire_new_buf(VmafFrameSyncContext *fs_ctx, void **data, unsigned data_sz, unsigned index);

int vmaf_framesync_submit_filled_data(VmafFrameSyncContext *fs_ctx, void *data, unsigned index);

int vmaf_framesync_retrive_filled_data(VmafFrameSyncContext *fs_ctx, void **data, unsigned index);

int vmaf_framesync_release_buf(VmafFrameSyncContext *fs_ctx, void *data, unsigned index);

int vmaf_framesync_destroy(VmafFrameSyncContext *fs_ctx);

#endif /* __VMAF_FRAME_SYNC_H__ */