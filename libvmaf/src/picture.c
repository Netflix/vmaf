/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "mem.h"
#include "picture.h"
#include "ref.h"

#define DATA_ALIGN 64

// Picture buffer pool: reuses freed pixel buffers to avoid mmap/munmap
// syscalls on each frame. Buffers are matched by exact size (LIFO stack).
// Thread-safe: vmaf_picture_unref can be called from worker threads.
#define PIC_POOL_MAX 8

static struct {
    struct { void *data; size_t size; } entries[PIC_POOL_MAX];
    unsigned count;
    pthread_mutex_t lock;
} pic_pool = { .lock = PTHREAD_MUTEX_INITIALIZER };

static void *pic_pool_acquire(size_t size)
{
    void *data = NULL;
    pthread_mutex_lock(&pic_pool.lock);
    for (unsigned i = pic_pool.count; i > 0; i--) {
        if (pic_pool.entries[i - 1].size == size) {
            data = pic_pool.entries[i - 1].data;
            pic_pool.entries[i - 1] = pic_pool.entries[--pic_pool.count];
            break;
        }
    }
    pthread_mutex_unlock(&pic_pool.lock);
    return data;
}

static void pic_pool_release(void *data, size_t size)
{
    pthread_mutex_lock(&pic_pool.lock);
    if (pic_pool.count < PIC_POOL_MAX) {
        pic_pool.entries[pic_pool.count].data = data;
        pic_pool.entries[pic_pool.count].size = size;
        pic_pool.count++;
        pthread_mutex_unlock(&pic_pool.lock);
        return;
    }
    pthread_mutex_unlock(&pic_pool.lock);
    aligned_free(data);  // pool full, discard
}

// Release callback that returns buffer to pool instead of freeing
static int pool_release_picture(VmafPicture *pic, void *cookie)
{
    size_t pic_size = (size_t)(uintptr_t)cookie;
    pic_pool_release(pic->data[0], pic_size);
    return 0;
}

static int default_release_picture(VmafPicture *pic, void *cookie)
{
    (void) cookie;
    aligned_free(pic->data[0]);
    return 0;
}

int vmaf_picture_set_release_callback(VmafPicture *pic, void *cookie,
                         int (*release_picture)(VmafPicture *pic, void *cookie))
{
    if (!pic) return -EINVAL;
    if (!release_picture) return -EINVAL;

    VmafPicturePrivate *priv = pic->priv;
    priv->cookie = cookie;
    priv->release_picture = release_picture;

    return 0;
}

int vmaf_picture_priv_init(VmafPicture *pic)
{
    const size_t priv_sz = sizeof(VmafPicturePrivate);
    pic->priv = malloc(priv_sz);
    if (!pic->priv) return -EINVAL;
    memset(pic->priv, 0, priv_sz);
    return 0;
}

/* NOLINTNEXTLINE(readability-function-size) */
int vmaf_picture_alloc(VmafPicture *pic, enum VmafPixelFormat pix_fmt,
                       unsigned bpc, unsigned w, unsigned h)
{
    if (!pic) return -EINVAL;
    if (!pix_fmt) return -EINVAL;
    if (bpc < 8 || bpc > 16) return -EINVAL;

    int err = 0;

    memset(pic, 0, sizeof(*pic));
    pic->pix_fmt = pix_fmt;
    pic->bpc = bpc;
    const int ss_hor = pic->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pic->pix_fmt == VMAF_PIX_FMT_YUV420P;
    pic->w[0] = w;
    pic->w[1] = pic->w[2] = w >> ss_hor;
    pic->h[0] = h;
    pic->h[1] = pic->h[2] = h >> ss_ver;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->w[1] = pic->w[2] = pic->h[1] = pic->h[2] = 0;

    const int aligned_y = (pic->w[0] + DATA_ALIGN - 1) & ~(DATA_ALIGN - 1);
    const int aligned_c = (pic->w[1] + DATA_ALIGN - 1) & ~(DATA_ALIGN - 1);
    const int hbd = pic->bpc > 8;
    pic->stride[0] = aligned_y << hbd;
    pic->stride[1] = pic->stride[2] = aligned_c << hbd;
    const size_t y_sz = pic->stride[0] * pic->h[0];
    const size_t uv_sz = pic->stride[1] * pic->h[1];
    const size_t pic_size = y_sz + 2 * uv_sz;

    // Try to reuse a buffer from the pool before allocating fresh
    uint8_t *data = pic_pool_acquire(pic_size);
    if (data) {
        memset(data, 0, pic_size);
    } else {
        data = aligned_malloc(pic_size, DATA_ALIGN);
        if (!data) goto fail;
        memset(data, 0, pic_size);
    }
    pic->data[0] = data;
    pic->data[1] = data + y_sz;
    pic->data[2] = data + y_sz + uv_sz;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->data[1] = pic->data[2] = NULL;

    err |= vmaf_picture_priv_init(pic);
    /* The callback userdata slot stores pic_size inline as a tagged value,
     * never dereferenced — standard uintptr_t idiom. */
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    err |= vmaf_picture_set_release_callback(pic, (void *)(uintptr_t)pic_size,
                                             pool_release_picture);
    if (err) goto free_data;

    err = vmaf_ref_init(&pic->ref);
    if (err) goto free_priv;

    return 0;

free_priv:
    free(pic->priv);
free_data:
    aligned_free(data);
fail:
    return -ENOMEM;
}

int vmaf_picture_ref(VmafPicture *dst, VmafPicture *src) {
    if (!dst || !src) return -EINVAL;

    memcpy(dst, src, sizeof(*src));
    vmaf_ref_fetch_increment(src->ref);
    return 0;
}

int vmaf_picture_unref(VmafPicture *pic) {
    if (!pic) return -EINVAL;
    if (!pic->ref) return -EINVAL;

    const long old_cnt = vmaf_ref_fetch_decrement(pic->ref);
    if (old_cnt == 1) {
        const VmafPicturePrivate *priv = pic->priv;
        priv->release_picture(pic, priv->cookie);
        free(pic->priv);
        vmaf_ref_close(pic->ref);
    }
    memset(pic, 0, sizeof(*pic));
    return 0;
}
