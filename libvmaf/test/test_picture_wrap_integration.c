/**
 *
 *  Copyright 2016-2025 Netflix, Inc.
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
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "picture.h"

typedef struct UserVideoDecoder {
    unsigned w, h, bpc;
    enum VmafPixelFormat pix_fmt;
    uint8_t **frame_buffers;
    unsigned frame_count;
} UserVideoDecoder;

static int user_decoder_init(UserVideoDecoder *dec, unsigned w, unsigned h,
                             unsigned bpc, enum VmafPixelFormat pix_fmt,
                             unsigned frame_count)
{
    dec->w = w;
    dec->h = h;
    dec->bpc = bpc;
    dec->pix_fmt = pix_fmt;
    dec->frame_count = frame_count;

    const unsigned ss_hor = pix_fmt != VMAF_PIX_FMT_YUV444P;
    const unsigned ss_ver = pix_fmt == VMAF_PIX_FMT_YUV420P;
    const unsigned w_c = w >> ss_hor;
    const unsigned h_c = h >> ss_ver;
    const int hbd = bpc > 8;

    const size_t y_sz = (w << hbd) * h;
    const size_t uv_sz = (w_c << hbd) * h_c;
    const size_t frame_sz = y_sz + 2 * uv_sz;

    dec->frame_buffers = malloc(sizeof(*dec->frame_buffers) * frame_count);
    if (!dec->frame_buffers) return -1;

    for (unsigned i = 0; i < frame_count; i++) {
        dec->frame_buffers[i] = malloc(frame_sz);
        if (!dec->frame_buffers[i]) {
            for (unsigned j = 0; j < i; j++) {
                free(dec->frame_buffers[j]);
            }
            free(dec->frame_buffers);
            return -1;
        }
        memset(dec->frame_buffers[i], i * 10, frame_sz);
    }

    return 0;
}

static void user_decoder_close(UserVideoDecoder *dec)
{
    if (!dec) return;
    if (!dec->frame_buffers) return;

    for (unsigned i = 0; i < dec->frame_count; i++)
        free(dec->frame_buffers[i]);
    free(dec->frame_buffers);
    dec->frame_buffers = NULL;
}

static int user_decoder_get_frame(UserVideoDecoder *dec, unsigned frame_idx,
                                  void *data[3], ptrdiff_t stride[3])
{
    if (frame_idx >= dec->frame_count) return -1;

    const unsigned ss_hor = dec->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const unsigned ss_ver = dec->pix_fmt == VMAF_PIX_FMT_YUV420P;
    const unsigned w_c = dec->w >> ss_hor;
    const unsigned h_c = dec->h >> ss_ver;
    const int hbd = dec->bpc > 8;

    stride[0] = dec->w << hbd;
    stride[1] = stride[2] = w_c << hbd;

    const size_t y_sz = stride[0] * dec->h;
    const size_t uv_sz = stride[1] * h_c;

    uint8_t *buf = dec->frame_buffers[frame_idx];
    data[0] = buf;
    data[1] = buf + y_sz;
    data[2] = buf + y_sz + uv_sz;

    return 0;
}

static char *test_picture_wrap_with_vmaf_zero_copy()
{
    int err = 0;

    UserVideoDecoder ref_dec, dist_dec;
    err = user_decoder_init(&ref_dec, 320, 240, 8, VMAF_PIX_FMT_YUV420P, 5);
    mu_assert("failed to init ref decoder", !err);
    err = user_decoder_init(&dist_dec, 320, 240, 8, VMAF_PIX_FMT_YUV420P, 5);
    mu_assert("failed to init dist decoder", !err);

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_INFO,
        .n_threads = 0,
    };
    VmafContext *vmaf;
    err = vmaf_init(&vmaf, cfg);
    mu_assert("problem during vmaf_init", !err);

    VmafModelConfig model_cfg = { 0 };
    VmafModel *model;
    err = vmaf_model_load(&model, &model_cfg, "vmaf_v0.6.1");
    mu_assert("problem during vmaf_model_load", !err);

    err = vmaf_use_features_from_model(vmaf, model);
    mu_assert("problem during vmaf_use_features_from_model", !err);

    for (unsigned i = 0; i < 5; i++) {
        void *ref_data[3], *dist_data[3];
        ptrdiff_t ref_stride[3], dist_stride[3];

        err = user_decoder_get_frame(&ref_dec, i, ref_data, ref_stride);
        mu_assert("failed to get ref frame", !err);
        err = user_decoder_get_frame(&dist_dec, i, dist_data, dist_stride);
        mu_assert("failed to get dist frame", !err);

        VmafPicture ref_pic, dist_pic;
        VmafPictureWrapped ref_wrapped = {
            .pix_fmt = VMAF_PIX_FMT_YUV420P, .bpc = 8, .w = 320, .h = 240,
            .data = { ref_data[0], ref_data[1], ref_data[2] },
            .stride = { ref_stride[0], ref_stride[1], ref_stride[2] },
        };
        VmafPictureWrapped dist_wrapped = {
            .pix_fmt = VMAF_PIX_FMT_YUV420P, .bpc = 8, .w = 320, .h = 240,
            .data = { dist_data[0], dist_data[1], dist_data[2] },
            .stride = { dist_stride[0], dist_stride[1], dist_stride[2] },
        };
        err = vmaf_picture_wrap(&ref_pic, ref_wrapped);
        mu_assert("problem during vmaf_picture_wrap (ref)", !err);

        err = vmaf_picture_wrap(&dist_pic, dist_wrapped);
        mu_assert("problem during vmaf_picture_wrap (dist)", !err);

        err = vmaf_read_pictures(vmaf, &ref_pic, &dist_pic, i);
        mu_assert("problem during vmaf_read_pictures", !err);

    }

    err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
    mu_assert("problem during vmaf_read_pictures flush", !err);

    double vmaf_score;
    err = vmaf_score_pooled(vmaf, model, VMAF_POOL_METHOD_MEAN, &vmaf_score, 0, 4);
    mu_assert("problem during vmaf_score_pooled", !err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);

    user_decoder_close(&ref_dec);
    user_decoder_close(&dist_dec);

    return NULL;
}

typedef struct UserFrameContext {
    unsigned frame_idx;
    int cleanup_called;
    void *buffer_ptr;
} UserFrameContext;

static int user_frame_cleanup(VmafPicture *pic, void *cookie)
{
    (void) pic;
    UserFrameContext *ctx = (UserFrameContext*)cookie;
    ctx->cleanup_called = 1;
    return 0;
}

static char *test_picture_wrap_with_cleanup_callback()
{
    int err = 0;

    const unsigned w = 320, h = 240;
    const size_t y_sz = w * h;
    const size_t uv_sz = (w/2) * (h/2);
    const size_t frame_sz = y_sz + 2 * uv_sz;

    uint8_t *user_buffer = malloc(frame_sz);
    mu_assert("failed to allocate user buffer", user_buffer != NULL);
    memset(user_buffer, 128, frame_sz);

    UserFrameContext ctx = {
        .frame_idx = 0,
        .cleanup_called = 0,
        .buffer_ptr = user_buffer
    };

    void *data[3] = {
        user_buffer,
        user_buffer + y_sz,
        user_buffer + y_sz + uv_sz
    };
    ptrdiff_t stride[3] = { w, w/2, w/2 };

    VmafPicture pic;
    VmafPictureWrapped pic_wrapped = {
        .pix_fmt = VMAF_PIX_FMT_YUV420P, .bpc = 8, .w = w, .h = h,
        .data = { data[0], data[1], data[2] },
        .stride = { stride[0], stride[1], stride[2] },
        .cookie = &ctx,
        .release_picture = user_frame_cleanup,
    };
    err = vmaf_picture_wrap(&pic, pic_wrapped);
    mu_assert("problem during vmaf_picture_wrap", !err);
    mu_assert("callback called too early", ctx.cleanup_called == 0);

    err = vmaf_picture_unref(&pic);
    mu_assert("problem during vmaf_picture_unref", !err);
    mu_assert("user cleanup callback not called", ctx.cleanup_called == 1);

    mu_assert("user buffer was corrupted", ((uint8_t*)ctx.buffer_ptr)[0] == 128);
    free(user_buffer);

    return NULL;
}

char *run_tests()
{
    mu_run_test(test_picture_wrap_with_vmaf_zero_copy);
    mu_run_test(test_picture_wrap_with_cleanup_callback);
    return NULL;
}
