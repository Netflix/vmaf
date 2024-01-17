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

#include <stdint.h>
#include <string.h>

#include "test.h"
#include "thread_pool.h"

#define NUM_TEST_FRAMES 10
#define FRAME_BUF_LEN   1024
#include "framesync.h"

typedef struct ThreadData {
    uint8_t *ref;
	uint8_t *dist;
    unsigned index;
	VmafFrameSyncContext *framesync;
    int err;
}ThreadData;

static void framesync_proc_func(void *frm_ctxt)
{
	int ctr;
    struct ThreadData *frame_ctxt = (ThreadData *)frm_ctxt;
	uint8_t *shared_buf;
	uint8_t *dependent_buf;
	
	//acquire new buffer from frame sync
	vmaf_framesync_aquire_new_buf(frame_ctxt->framesync,(void **)&shared_buf,FRAME_BUF_LEN,frame_ctxt->index);
	
	//populate shared buffer with values
	for(ctr = 0; ctr < FRAME_BUF_LEN; ctr++)
	{
		shared_buf[ctr] = frame_ctxt->ref[ctr] + frame_ctxt->dist[ctr] + 2 ;
	}
	
	// submit filled buffer back to frame sync
	vmaf_framesync_submit_filled_data(frame_ctxt->framesync,shared_buf,frame_ctxt->index);
	
	//sleep to similuate work load
	sleep(1);
	
	
	if(frame_ctxt->index !=0)
	{
		//retrive dependent buffer from frame sync
		vmaf_framesync_retrive_filled_data(frame_ctxt->framesync,(void **)&dependent_buf,(frame_ctxt->index-1));
		
		for(ctr = 0; ctr < FRAME_BUF_LEN; ctr++)
		{
			if(dependent_buf[ctr] != (frame_ctxt->ref[ctr] + frame_ctxt->dist[ctr]))
				printf("Verification Error in frame index %d\n",frame_ctxt->index);
		}
		//release dependent buffer from frame sync
		vmaf_framesync_release_buf (frame_ctxt->framesync,dependent_buf,(frame_ctxt->index-1));
	}


	free(frame_ctxt->ref);
	free(frame_ctxt->dist);
}


static char *test_framesync_create_process_and_destroy()
{
    int err, frame_index;

    VmafThreadPool *pool;
	VmafFrameSyncContext *framesync;
    unsigned n_threads = 2;

    err = vmaf_thread_pool_create(&pool, n_threads);
    mu_assert("problem during vmaf_thread_pool_init", !err);

	err = vmaf_framesync_init(&framesync);
	mu_assert("problem during vmaf_framesync_init", !err);
	
	//loop over frames to be tested
	for(frame_index = 0; frame_index <NUM_TEST_FRAMES; frame_index++)
	{
		uint8_t *pic_a = malloc(FRAME_BUF_LEN);
		uint8_t *pic_b = malloc(FRAME_BUF_LEN);

		printf("Processing frame %d\n\n", frame_index);
		
		memset(pic_a, frame_index, FRAME_BUF_LEN);
		memset(pic_b, (frame_index), FRAME_BUF_LEN);
		
		struct ThreadData data = {
            .ref = pic_a,
            .dist = pic_b,
            .index = frame_index,
            .framesync = framesync,
            .err = 0,
    };

		err = vmaf_thread_pool_enqueue(pool, framesync_proc_func, &data, sizeof(ThreadData));

		mu_assert("problem during vmaf_thread_pool_enqueue with data", !err);

		//wait once in 2 frames
		if((frame_index >=1) && (frame_index & 1))
		{
			err = vmaf_thread_pool_wait(pool);
			mu_assert("problem during vmaf_thread_pool_wait", !err);
		}
	}
    err = vmaf_thread_pool_wait(pool);
    mu_assert("problem during vmaf_thread_pool_wait", !err);
    err = vmaf_thread_pool_destroy(pool);
    mu_assert("problem during vmaf_thread_pool_destroy", !err);
	err = vmaf_framesync_destroy(framesync);
	mu_assert("problem during vmaf_framesync_destroy", !err);
    printf("\n");

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_framesync_create_process_and_destroy);
    return NULL;
}
