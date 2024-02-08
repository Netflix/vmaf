/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2022 NVIDIA Corporation.
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
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "ring_buffer.h"

#ifdef HAVE_NVTX
#include "nvtx3/nvToolsExt.h"
#endif

typedef struct VmafRingBuffer {
    VmafRingBufferConfig cfg;
    unsigned curr_idx;
    pthread_mutex_t busy;
    VmafPicture *pic;
} VmafRingBuffer;

int vmaf_ring_buffer_init(VmafRingBuffer **ring_buffer,
                          VmafRingBufferConfig cfg)
{
    if (!ring_buffer) return -EINVAL;
    if (!cfg.pic_cnt) return -EINVAL;
    if (!cfg.alloc_picture_callback) return -EINVAL;
    if (!cfg.free_picture_callback) return -EINVAL;

    int err = 0;

    VmafRingBuffer *const rb = *ring_buffer = malloc(sizeof(*rb));
    if (!rb) goto fail;
    memset(rb, 0, sizeof(*rb));
    rb->cfg = cfg;

    rb->pic = malloc(sizeof(VmafPicture) * rb->cfg.pic_cnt);
    if (!rb->pic) {
        err = -ENOMEM;
        goto free_rb;
    }

    err = pthread_mutex_init(&rb->busy, NULL);
    if (err) goto free_pic;

    for (unsigned i = 0; i < rb->cfg.pic_cnt; i++)
        err |= rb->cfg.alloc_picture_callback(&rb->pic[i], rb->cfg.cookie);

    return err;

free_pic:
    free(rb->pic);
free_rb:
    free(rb);
fail:
    return err;
}

int vmaf_ring_buffer_close(VmafRingBuffer *ring_buffer)
{
    if (!ring_buffer) return -EINVAL;

    int err = pthread_mutex_lock(&ring_buffer->busy);
    if (err) return err;

    for (unsigned i = 0; i < ring_buffer->cfg.pic_cnt; i++) {
        err |= ring_buffer->cfg.free_picture_callback(&ring_buffer->pic[i],
                                                      ring_buffer->cfg.cookie);
    }

    free(ring_buffer->pic);
    return err;
}

int vmaf_ring_buffer_fetch_next_picture(VmafRingBuffer *ring_buffer,
                                        VmafPicture *pic)
{
    if (!ring_buffer) return -EINVAL;
    if (!pic) return -EINVAL;

    int err = pthread_mutex_lock(&ring_buffer->busy);
    if (err) return err;
    unsigned pic_idx = ring_buffer->curr_idx;
    ring_buffer->curr_idx = (ring_buffer->curr_idx+1) % ring_buffer->cfg.pic_cnt;
    err |= pthread_mutex_unlock(&ring_buffer->busy);
    if (err) return err;

#ifdef HAVE_NVTX
    char n[40];
    static unsigned glob = 0;
    sprintf(n, "fetch idx %d %d", pic_idx, glob++);
    nvtxRangePushA(n);
#endif

    vmaf_picture_ref(pic, &ring_buffer->pic[pic_idx]);

    if (ring_buffer->cfg.synchronize_picture_callback) {
        err |= ring_buffer->cfg.synchronize_picture_callback(pic,
                ring_buffer->cfg.cookie);
    }

#ifdef HAVE_NVTX
    nvtxRangePop();
#endif

    return err;
}
