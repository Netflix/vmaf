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

#include "test.h"

#include "libvmaf/vmaf_cuda.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"

static char *test_cuda_no_init()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

static char *test_cuda_picture_preallocation_method_none()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaConfiguration cuda_cfg = {
        .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_NONE,
    };

    VmafCudaState *cu_state;
    err = vmaf_cuda_init(vmaf, &cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_init", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, 1920, 1080);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

static char *test_cuda_picture_preallocation_method_host()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaConfiguration cuda_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST,
    };

    VmafCudaState *cu_state;
    err = vmaf_cuda_init(vmaf, &cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_init", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &ref);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &dist);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

static char *test_cuda_picture_preallocation_method_host_pinned()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaConfiguration cuda_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_HOST_PINNED,
    };

    VmafCudaState *cu_state;
    err = vmaf_cuda_init(vmaf, &cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_init", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &ref);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &dist);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

static char *test_cuda_picture_preallocation_method_device()
{
    int err = 0;

    VmafConfiguration vmaf_cfg = { 0 };

    VmafContext *vmaf;
    vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", vmaf);

    VmafCudaConfiguration cuda_cfg = {
        .pic_params = {
            .w = 1920,
            .h = 1080,
            .bpc = 8,
            .pix_fmt = VMAF_PIX_FMT_YUV420P,
        },
        .pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE,
    };

    VmafCudaState *cu_state;
    err = vmaf_cuda_init(vmaf, &cu_state, cuda_cfg);
    mu_assert("problem during vmaf_cuda_init", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", model);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 10; i++) {
        VmafPicture ref, dist;
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &ref);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_cuda_fetch_preallocated_picture(vmaf, &dist);
        mu_assert("problem during vmaf_picture_alloc", !err);
        err = vmaf_read_pictures(vmaf, &ref, &dist, i);
        mu_assert("problem during vmaf_read_pictures", !err);
    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_read_pictures", !err);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_cuda_no_init);
    mu_run_test(test_cuda_picture_preallocation_method_none);
    mu_run_test(test_cuda_picture_preallocation_method_host);
    mu_run_test(test_cuda_picture_preallocation_method_host_pinned);
    mu_run_test(test_cuda_picture_preallocation_method_device);
    return NULL;
}
