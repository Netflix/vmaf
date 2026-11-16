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
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "test.h"

#include "libvmaf/libvmaf_cuda.h"

#include "cuda/common.h"

static char *test_cuda_buffer_alloc_arguments()
{
    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = { 0 };
    int err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_state_init", !err);

    VmafCudaBuffer *buf = NULL;
    err = vmaf_cuda_buffer_alloc(cu_state, NULL, 1024);
    mu_assert("a NULL buffer pointer must give -EINVAL", err == -EINVAL);
    err = vmaf_cuda_buffer_alloc(NULL, &buf, 1024);
    mu_assert("a NULL state must give -EINVAL", err == -EINVAL);
    mu_assert("buffer must stay NULL after an argument error", !buf);

    err = vmaf_cuda_release(cu_state);
    mu_assert("problem during vmaf_cuda_release", !err);
    return NULL;
}

static char *test_cuda_buffer_alloc_out_of_memory()
{
    VmafCudaState *cu_state;
    VmafCudaConfiguration cuda_cfg = { 0 };
    int err = vmaf_cuda_state_init(&cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_state_init", !err);

    /* a request no device can satisfy: it must fail with an error code, not abort */
    VmafCudaBuffer *buf = NULL;
    err = vmaf_cuda_buffer_alloc(cu_state, &buf, SIZE_MAX / 2);
    mu_assert("an impossible allocation must fail with -ENOMEM", err == -ENOMEM);
    mu_assert("buffer must stay NULL after a failed allocation", !buf);

    /* the state is still usable afterwards: the context was popped */
    err = vmaf_cuda_buffer_alloc(cu_state, &buf, 1 << 20);
    mu_assert("allocation after a failed one must succeed", !err);
    mu_assert("buffer was not returned", buf);
    mu_assert("buffer has the requested size", buf->size == (1 << 20));
    err = vmaf_cuda_buffer_free(cu_state, buf);
    mu_assert("problem during vmaf_cuda_buffer_free", !err);
    free(buf);

    err = vmaf_cuda_release(cu_state);
    mu_assert("problem during vmaf_cuda_release", !err);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_cuda_buffer_alloc_arguments);
    mu_run_test(test_cuda_buffer_alloc_out_of_memory);
    return NULL;
}
