#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "picture.h"
#include "ref.h"

typedef struct VmafPicturePool {
    VmafPicturePoolConfig cfg;
    pthread_mutex_t lock;
    VmafPicture *pic;
    struct {
        VmafPicture *head, *tail;
        pthread_cond_t available;
        pthread_cond_t empty;
    } queue;
} VmafPicturePool;

static int release_picture_callback(VmafPicture *pic, void *cookie)
{
    if (!pic) return -EINVAL;

    int err = 0;
    VmafPicturePool *pic_pool = cookie;

    if (pic_pool->cfg.synchronize_picture_callback)
        pic_pool->cfg.synchronize_picture_callback(pic, cookie);

    VmafPicture pic_copy;
    memcpy(&pic_copy, pic, sizeof(pic_copy));

    err |= vmaf_ref_init(&pic_copy.ref);
    err |= vmaf_picture_priv_init(&pic_copy);
    err |= vmaf_picture_set_release_callback(&pic_copy, pic_pool,
                                             release_picture_callback);

    pthread_mutex_lock(&(pic_pool->lock));
    pic_pool->queue.head--;
    memcpy(pic_pool->queue.head, &pic_copy, sizeof(pic_copy));
    pthread_cond_signal(&(pic_pool->queue.available));
    if (pic_pool->queue.head == pic_pool->pic)
        pthread_cond_signal(&(pic_pool->queue.empty));
    pthread_mutex_unlock(&(pic_pool->lock));

    return err;
}

int vmaf_picture_pool_init(VmafPicturePool **pic_pool,
                           VmafPicturePoolConfig cfg)
{
    if (!pic_pool) return -EINVAL;
    if (!cfg.pic_cnt) return -EINVAL;
    if (!cfg.alloc_picture_callback) return -EINVAL;
    if (!cfg.free_picture_callback) return -EINVAL;

    int err = 0;

    VmafPicturePool *const p = *pic_pool = malloc(sizeof(*p));
    if (!p) goto fail;
    memset(p, 0, sizeof(*p));
    p->cfg = cfg;

    const size_t pic_data_sz =
        sizeof(VmafPicture) * p->cfg.pic_cnt + 1;
    p->pic = malloc(pic_data_sz);
    if (!p->pic) goto free_p;
    memset(p->pic, 0, pic_data_sz);

    p->queue.head = &p->pic[0];
    p->queue.tail = &p->pic[p->cfg.pic_cnt];

    pthread_mutex_init(&(p->lock), NULL);
    pthread_mutex_lock(&(p->lock));
    pthread_cond_init(&(p->queue.available), NULL);
    pthread_cond_init(&(p->queue.empty), NULL);

    for (unsigned i = 0; i < p->cfg.pic_cnt; i++) {
        VmafPicture *pic = &p->pic[i];
        err |= p->cfg.alloc_picture_callback(pic, p->cfg.cookie);

        err |= vmaf_ref_init(&pic->ref);
        err |= vmaf_picture_priv_init(pic);
        err |= vmaf_picture_set_release_callback(pic, p,
                                                 release_picture_callback);
    }

    pthread_mutex_unlock(&(p->lock));
    return err;

free_p:
    free(p);
fail:
    return -ENOMEM;
}

int vmaf_picture_pool_request_picture(VmafPicturePool *pic_pool,
                                      VmafPicture *pic)
{
    if (!pic_pool) return -EINVAL;
    if (!pic) return -EINVAL;

    int err = 0;
    pthread_mutex_lock(&(pic_pool->lock));

    while (pic_pool->queue.head == pic_pool->queue.tail)
        pthread_cond_wait(&(pic_pool->queue.available), &(pic_pool->lock));

    *pic = *pic_pool->queue.head;
    pic_pool->queue.head++;

    pthread_mutex_unlock(&(pic_pool->lock));
    return err;
}

int vmaf_picture_pool_close(VmafPicturePool *pic_pool)
{
    if (!pic_pool) return -EINVAL;

    int err = 0;

    pthread_mutex_lock(&(pic_pool->lock));

    while (pic_pool->queue.head != pic_pool->pic)
        pthread_cond_wait(&(pic_pool->queue.empty), &(pic_pool->lock));

    for (unsigned i = 0; i < pic_pool->cfg.pic_cnt; i++) {
        err |= pic_pool->cfg.free_picture_callback(&pic_pool->pic[i],
                                                   pic_pool->cfg.cookie);
    }

    pthread_cond_destroy(&(pic_pool->queue.empty));
    pthread_cond_destroy(&(pic_pool->queue.available));
    pthread_mutex_destroy(&(pic_pool->lock));
    free(pic_pool);

    return err;
}
