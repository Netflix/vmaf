/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *
 *  Licensed under the BSD+Patent License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      https://opensource.org/licenses/BSDplusPatent
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

/**
 * @file vmaf_dnn_stub.c
 * @brief Disabled-build stub for the DNN surface.
 *
 * Compiled when `-Denable_dnn=false` (default). Every entry point returns
 * -ENOSYS so the feature extractor can degrade gracefully.
 */

#include <errno.h>

#include "vmaf_dnn.h"

int vmaf_dnn_open(VmafDnnSession **out, const char *path,
                  VmafDnnProvider provider)
{
    (void) out;
    (void) path;
    (void) provider;
    return -ENOSYS;
}

int vmaf_dnn_infer(VmafDnnSession *sess, const float *features,
                   size_t n_features, double *score)
{
    (void) sess;
    (void) features;
    (void) n_features;
    (void) score;
    return -ENOSYS;
}

void vmaf_dnn_close(VmafDnnSession *sess)
{
    (void) sess;
}
