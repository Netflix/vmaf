
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

#include <errno.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include "framesync.h"

int vmaf_framesync_init(VmafFrameSyncContext **fs_ctx)
{
	VmafFrameSyncContext *const ctx = *fs_ctx = malloc(sizeof(VmafFrameSyncContext));
    if (!ctx) return -ENOMEM;
    memset(ctx, 0, sizeof(VmafFrameSyncContext));
    ctx->buf_cnt = MAX_FRAME_SYNC_BUF_Q_LEN;
	
	pthread_mutex_init(&(ctx->aquire_lock), NULL);
	pthread_mutex_init(&(ctx->retrive_lock), NULL);
    pthread_cond_init(&(ctx->aquire), NULL);
    pthread_cond_init(&(ctx->retrive), NULL);
	
	VmafFrameSyncBuff *buf_que = ctx->buf_que;
	
	for (unsigned i = 0; i < ctx->buf_cnt; i++) {
        buf_que[i].frame_data = NULL;
		buf_que[i].buf_status = BUFF_FREE;
		buf_que[i].index = -1;
	}
	
    return 0;
}

int vmaf_framesync_aquire_new_buf(VmafFrameSyncContext *fs_ctx, void **data, unsigned data_sz, unsigned index)
{
	VmafFrameSyncBuff *buf_que = fs_ctx->buf_que;
	*data = NULL;
	
    
	while (*data == NULL)
	{
		pthread_mutex_lock(&(fs_ctx->aquire_lock));
		/* loop unitl a free buffer is found */
		for (unsigned i = 0; i < fs_ctx->buf_cnt; i++)
		{
			if(buf_que[i].buf_status == BUFF_FREE)
			{
				buf_que[i].buf_status = BUFF_AQUIRED;
				buf_que[i].index = index;
				buf_que[i].frame_data = *data = malloc(data_sz);
				if (!buf_que[i].frame_data) return -ENOMEM;
				break;
			}
		}  
		
		// check for successful new buffer allocation
		if(*data == NULL)
		{
			pthread_cond_wait(&(fs_ctx->aquire), &(fs_ctx->aquire_lock));
		}
		pthread_mutex_unlock(&(fs_ctx->aquire_lock));
	}    
	
	return 0;
}

int vmaf_framesync_submit_filled_data(VmafFrameSyncContext *fs_ctx, void *data, unsigned index)
{
	VmafFrameSyncBuff *buf_que = fs_ctx->buf_que;
	
	pthread_mutex_lock(&(fs_ctx->retrive_lock));
	/* loop unitl a matchng buffer is found */
	for (unsigned i = 0; i < fs_ctx->buf_cnt; i++)
	{
		if((buf_que[i].index == index) && (buf_que[i].buf_status == BUFF_AQUIRED))
		{
			buf_que[i].buf_status = BUFF_FILLED;
			if (data != buf_que[i].frame_data){
				return -1;
			}
			break;
		}
	} 

	pthread_cond_broadcast(&(fs_ctx->retrive));	
	pthread_mutex_unlock(&(fs_ctx->retrive_lock));
	
    return 0;
}

int vmaf_framesync_retrive_filled_data(VmafFrameSyncContext *fs_ctx, void **data, unsigned index)
{
    VmafFrameSyncBuff *buf_que = fs_ctx->buf_que;
	*data = NULL;
	    
	while (*data == NULL)
	{
		pthread_mutex_lock(&(fs_ctx->retrive_lock));
		/* loop unitl a free buffer is found */
		for (unsigned i = 0; i < fs_ctx->buf_cnt; i++)
		{
			if((buf_que[i].index == index) && (buf_que[i].buf_status == BUFF_FILLED))
			{
				buf_que[i].buf_status = BUFF_RETRIVED;
				*data = buf_que[i].frame_data;
				break;
			}
		}  
		
		if(*data == NULL)
		{
			pthread_cond_wait(&(fs_ctx->retrive), &(fs_ctx->retrive_lock));
		}
		pthread_mutex_unlock(&(fs_ctx->retrive_lock));
	}    
	
    return 0;
}

int vmaf_framesync_release_buf(VmafFrameSyncContext *fs_ctx, void *data, unsigned index)
{
	VmafFrameSyncBuff *buf_que = fs_ctx->buf_que;
	
	pthread_mutex_lock(&(fs_ctx->aquire_lock));
	/* loop unitl a matchng buffer is found */
	for (unsigned i = 0; i < fs_ctx->buf_cnt; i++)
	{
		if((buf_que[i].index == index) && (buf_que[i].buf_status == BUFF_RETRIVED))
		{
			if (data != buf_que[i].frame_data){
				return -1;
			}
			free(buf_que[i].frame_data);
			buf_que[i].frame_data = NULL;
			buf_que[i].buf_status = BUFF_FREE;
			buf_que[i].index = -1;
			break;
		}
	} 

	pthread_cond_broadcast(&(fs_ctx->aquire));
	pthread_mutex_unlock(&(fs_ctx->aquire_lock));
    return 0;
}

int vmaf_framesync_destroy(VmafFrameSyncContext *fs_ctx)
{
	pthread_mutex_destroy(&(fs_ctx->aquire_lock));
	pthread_mutex_destroy(&(fs_ctx->retrive_lock));
    pthread_cond_destroy(&(fs_ctx->aquire));
    pthread_cond_destroy(&(fs_ctx->retrive));
	
	//check for any data buffers which are not freed
	for (unsigned i = 0; i < fs_ctx->buf_cnt; i++) {
		if(NULL != fs_ctx->buf_que[i].frame_data)
		{
			free(fs_ctx->buf_que[i].frame_data);
			fs_ctx->buf_que[i].frame_data = NULL;
		}
	}
		
	free(fs_ctx);
	
    return 0;
}
