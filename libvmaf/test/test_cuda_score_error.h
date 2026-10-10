/**
 *
 *  Copyright 2026 Bardie Høgh Joensen
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

#ifndef TEST_CUDA_SCORE_ERROR_H
#define TEST_CUDA_SCORE_ERROR_H

/* A duplicate score causes a collector error inside the CUDA host callback.
 * Exercise both pending final callbacks and callback-slot reuse. */
static char *test_cuda_score_error(const char *feature, const char *key)
{
    for (unsigned frames = 1; frames <= 3; frames += 2) {
        VmafContext *vmaf;
        int err = vmaf_init(&vmaf, (VmafConfiguration){ .log_level = VMAF_LOG_LEVEL_NONE });
        mu_assert("error-test context initialization failed", !err);
        VmafCudaState *state;
        err = vmaf_cuda_state_init(&state, (VmafCudaConfiguration){ 0 });
        mu_assert("error-test CUDA initialization failed", !err);
        err = vmaf_cuda_import_state(vmaf, state);
        mu_assert("error-test CUDA import failed", !err);
        err = vmaf_use_feature(vmaf, feature, NULL);
        mu_assert("error-test feature registration failed", !err);
        err = vmaf_import_feature_score(vmaf, key, -999.0, 0);
        mu_assert("could not seed duplicate feature score", !err);
        int extract_err = 0;
        for (unsigned i = 0; i < frames; i++) {
            VmafPicture ref, dist;
            err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV444P, 8, 16, 16);
            mu_assert("error-test reference allocation failed", !err);
            err = vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV444P, 8, 16, 16);
            mu_assert("error-test distorted allocation failed", !err);
            for (unsigned p = 0; p < 3; p++) {
                memset(ref.data[p], 100, ref.stride[p] * ref.h[p]);
                memset(dist.data[p], 101, dist.stride[p] * dist.h[p]);
            }
            extract_err = vmaf_read_pictures(vmaf, &ref, &dist, i);
            if (extract_err) {
                vmaf_picture_unref(&ref);
                vmaf_picture_unref(&dist);
                break;
            }
        }
        const int flush_err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
        vmaf_close(vmaf);
        cuda_free_functions(&state->f);
        free(state);
        mu_assert("pending CUDA callback error was lost during flush", flush_err < 0);
        if (frames == 3)
            mu_assert("CUDA callback error was lost when reusing a slot", extract_err < 0);
    }
    return NULL;
}

#endif
