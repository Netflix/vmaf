#ifndef __VMAF_THREAD_POOL_H__
#define __VMAF_THREAD_POOL_H__

#include <pthread.h>

typedef struct VmafThreadPool VmafThreadPool;

int vmaf_thread_pool_create(VmafThreadPool **tpool, unsigned n_threads);

int vmaf_thread_pool_enqueue(VmafThreadPool *pool, void (*func)(void *data),
                             void *data, size_t data_sz);

int vmaf_thread_pool_wait(VmafThreadPool *pool);

int vmaf_thread_pool_destroy(VmafThreadPool *tpool);

#endif /* __VMAF_THREAD_POOL_H__ */
