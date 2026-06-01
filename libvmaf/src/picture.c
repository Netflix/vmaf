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

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_ZIMG
#include <stdbool.h>
#include <zimg.h>
#endif

#include "log.h"
#include "mem.h"
#include "picture.h"
#include "ref.h"

#define DATA_ALIGN 32

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

static int wrap_release_picture(VmafPicture *pic, void *cookie)
{
    (void) pic;
    (void) cookie;
    return 0;
}

int vmaf_picture_wrap(VmafPicture *pic,
                      VmafPictureWrapped pic_wrapped)
{
    if (!pic) return -EINVAL;
    if (!pic_wrapped.pix_fmt) return -EINVAL;
    if (pic_wrapped.bpc < 8 || pic_wrapped.bpc > 16) return -EINVAL;

    int err = 0;

    memset(pic, 0, sizeof(*pic));
    pic->pix_fmt = pic_wrapped.pix_fmt;
    pic->bpc = pic_wrapped.bpc;
    const int ss_hor = pic->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = pic->pix_fmt == VMAF_PIX_FMT_YUV420P;
    pic->w[0] = pic_wrapped.w;
    pic->w[1] = pic->w[2] = pic_wrapped.w >> ss_hor;
    pic->h[0] = pic_wrapped.h;
    pic->h[1] = pic->h[2] = pic_wrapped.h >> ss_ver;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->w[1] = pic->w[2] = pic->h[1] = pic->h[2] = 0;

    for (int i = 0; i < 3; i++) {
        pic->data[i] = pic_wrapped.data[i];
        pic->stride[i] = pic_wrapped.stride[i];
    }

    err |= vmaf_picture_priv_init(pic);
    err |= vmaf_picture_set_release_callback(pic, pic_wrapped.cookie,
               pic_wrapped.release_picture ? pic_wrapped.release_picture : wrap_release_picture);
    if (err) goto free_priv;

    err = vmaf_ref_init(&pic->ref);
    if (err) goto free_priv;

    return 0;

free_priv:
    free(pic->priv);
    return -ENOMEM;
}

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

    uint8_t *data = aligned_malloc(pic_size, DATA_ALIGN);
    if (!data) goto fail;
    memset(data, 0, pic_size);
    pic->data[0] = data;
    pic->data[1] = data + y_sz;
    pic->data[2] = data + y_sz + uv_sz;
    if (pic->pix_fmt == VMAF_PIX_FMT_YUV400P)
        pic->data[1] = pic->data[2] = NULL;

    err |= vmaf_picture_priv_init(pic);
    err |= vmaf_picture_set_release_callback(pic, NULL, default_release_picture);
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

#ifdef HAVE_ZIMG

static int pix_fmt_to_zimg(enum VmafPixelFormat pix_fmt,
                           zimg_color_family_e *color_family,
                           unsigned *subsample_w, unsigned *subsample_h)
{
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV400P:
        *color_family = ZIMG_COLOR_GREY;
        *subsample_w = 0;
        *subsample_h = 0;
        return 0;
    case VMAF_PIX_FMT_YUV420P:
        *color_family = ZIMG_COLOR_YUV;
        *subsample_w = 1;
        *subsample_h = 1;
        return 0;
    case VMAF_PIX_FMT_YUV422P:
        *color_family = ZIMG_COLOR_YUV;
        *subsample_w = 1;
        *subsample_h = 0;
        return 0;
    case VMAF_PIX_FMT_YUV444P:
        *color_family = ZIMG_COLOR_YUV;
        *subsample_w = 0;
        *subsample_h = 0;
        return 0;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: unsupported pixel format %d "
                 "(supported: YUV400P, YUV420P, YUV422P, YUV444P)\n",
                 (int) pix_fmt);
        return -EINVAL;
    }
}

static int matrix_to_zimg(enum VmafColorMatrixCoefficients matrix,
                          zimg_matrix_coefficients_e *zimg_matrix)
{
    switch (matrix) {
    case VMAF_COLOR_MATRIX_BT709:
        *zimg_matrix = ZIMG_MATRIX_709;
        return 0;
    case VMAF_COLOR_MATRIX_BT2020_NCL:
        *zimg_matrix = ZIMG_MATRIX_2020_NCL;
        return 0;
    case VMAF_COLOR_MATRIX_ICTCP:
        *zimg_matrix = ZIMG_MATRIX_ICTCP;
        return 0;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: unsupported color matrix %d "
                 "(supported: BT709, BT2020_NCL, ICTCP)\n", (int) matrix);
        return -EINVAL;
    }
}

static int trc_to_zimg(enum VmafColorTransferCharacteristic trc,
                       zimg_transfer_characteristics_e *zimg_trc)
{
    switch (trc) {
    case VMAF_COLOR_TRC_BT709:
        *zimg_trc = ZIMG_TRANSFER_709;
        return 0;
    case VMAF_COLOR_TRC_SMPTE2084:
        *zimg_trc = ZIMG_TRANSFER_ST2084;
        return 0;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: unsupported transfer characteristic "
                 "%d (supported: BT709, SMPTE2084)\n", (int) trc);
        return -EINVAL;
    }
}

static int primaries_to_zimg(enum VmafColorPrimaries primaries,
                             zimg_color_primaries_e *zimg_primaries)
{
    switch (primaries) {
    case VMAF_COLOR_PRIMARIES_BT709:
        *zimg_primaries = ZIMG_PRIMARIES_709;
        return 0;
    case VMAF_COLOR_PRIMARIES_BT2020:
        *zimg_primaries = ZIMG_PRIMARIES_2020;
        return 0;
    case VMAF_COLOR_PRIMARIES_SMPTE432:
        *zimg_primaries = ZIMG_PRIMARIES_ST432_1;
        return 0;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: unsupported color primaries %d "
                 "(supported: BT709, BT2020, SMPTE432)\n", (int) primaries);
        return -EINVAL;
    }
}

static int range_to_zimg(enum VmafColorRange range,
                         zimg_pixel_range_e *zimg_range)
{
    switch (range) {
    case VMAF_COLOR_RANGE_LIMITED:
        *zimg_range = ZIMG_RANGE_LIMITED;
        return 0;
    case VMAF_COLOR_RANGE_FULL:
        *zimg_range = ZIMG_RANGE_FULL;
        return 0;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: unsupported color range %d "
                 "(supported: LIMITED, FULL)\n", (int) range);
        return -EINVAL;
    }
}

static int format_to_zimg(zimg_image_format *fmt,
                          enum VmafPixelFormat pix_fmt, unsigned bpc,
                          unsigned w, unsigned h, const VmafColor *color)
{
    zimg_color_family_e color_family;
    unsigned subsample_w, subsample_h;
    int err = pix_fmt_to_zimg(pix_fmt, &color_family,
                              &subsample_w, &subsample_h);
    if (err) return err;

    zimg_matrix_coefficients_e matrix;
    zimg_transfer_characteristics_e trc;
    zimg_color_primaries_e primaries;
    zimg_pixel_range_e range;
    err = matrix_to_zimg(color->matrix, &matrix);
    if (err) return err;
    err = trc_to_zimg(color->trc, &trc);
    if (err) return err;
    err = primaries_to_zimg(color->primaries, &primaries);
    if (err) return err;
    err = range_to_zimg(color->range, &range);
    if (err) return err;

    zimg_image_format_default(fmt, ZIMG_API_VERSION);
    fmt->width = w;
    fmt->height = h;
    fmt->pixel_type = bpc > 8 ? ZIMG_PIXEL_WORD : ZIMG_PIXEL_BYTE;
    fmt->depth = bpc;
    fmt->color_family = color_family;
    fmt->subsample_w = subsample_w;
    fmt->subsample_h = subsample_h;
    fmt->matrix_coefficients = matrix;
    fmt->transfer_characteristics = trc;
    fmt->color_primaries = primaries;
    fmt->pixel_range = range;
    return 0;
}

static zimg_resample_filter_e resample_filter_to_zimg(enum VmafResampleFilter f)
{
    switch (f) {
    case VMAF_RESAMPLE_BILINEAR: return ZIMG_RESIZE_BILINEAR;
    case VMAF_RESAMPLE_BICUBIC: return ZIMG_RESIZE_BICUBIC;
    case VMAF_RESAMPLE_LANCZOS: return ZIMG_RESIZE_LANCZOS;
    default: return ZIMG_RESIZE_BICUBIC;
    }
}

static unsigned n_planes(const VmafPicture *pic)
{
    return pic->pix_fmt == VMAF_PIX_FMT_YUV400P ? 1 : 3;
}

static void image_buffer_const_from_picture(zimg_image_buffer_const *buf,
                                            const VmafPicture *pic)
{
    memset(buf, 0, sizeof(*buf));
    buf->version = ZIMG_API_VERSION;
    const unsigned planes = n_planes(pic);
    for (unsigned i = 0; i < planes; i++) {
        buf->plane[i].data = pic->data[i];
        buf->plane[i].stride = pic->stride[i];
        buf->plane[i].mask = ZIMG_BUFFER_MAX;
    }
}

static void image_buffer_from_picture(zimg_image_buffer *buf,
                                      const VmafPicture *pic)
{
    memset(buf, 0, sizeof(*buf));
    buf->version = ZIMG_API_VERSION;
    const unsigned planes = n_planes(pic);
    for (unsigned i = 0; i < planes; i++) {
        buf->plane[i].data = pic->data[i];
        buf->plane[i].stride = pic->stride[i];
        buf->plane[i].mask = ZIMG_BUFFER_MAX;
    }
}

static bool color_is_specified(const VmafColor *color)
{
    return color->range != VMAF_COLOR_RANGE_UNKNOWN &&
           color->primaries != VMAF_COLOR_PRIMARIES_UNKNOWN &&
           color->trc != VMAF_COLOR_TRC_UNKNOWN &&
           color->matrix != VMAF_COLOR_MATRIX_UNKNOWN;
}

struct VmafPictureConvertContext {
    zimg_filter_graph *graph;
    void *tmp;
    VmafPictureConvertTarget target;
    struct {
        enum VmafPixelFormat pix_fmt;
        unsigned bpc;
        unsigned w, h;
        VmafColor color;
    } src;
};

static bool color_equal(const VmafColor *a, const VmafColor *b)
{
    return a->range == b->range && a->primaries == b->primaries &&
           a->trc == b->trc && a->matrix == b->matrix;
}

int vmaf_picture_convert_context_init(VmafPictureConvertContext **ctx,
                                      const VmafPicture *src,
                                      const VmafPictureConvertTarget *target)
{
    if (!ctx || !src || !target) return -EINVAL;

    if (!color_is_specified(&src->color)) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: source picture color "
                 "metadata must be fully specified\n");
        return -EINVAL;
    }

    if (!color_is_specified(&target->color)) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: target color "
                 "metadata must be fully specified\n");
        return -EINVAL;
    }

    if (target->bpc < 8 || target->bpc > 16) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: unsupported target "
                 "bit depth %u (supported: 8 to 16)\n", target->bpc);
        return -EINVAL;
    }

    const unsigned dst_w = target->w ? target->w : src->w[0];
    const unsigned dst_h = target->h ? target->h : src->h[0];

    zimg_image_format src_fmt, dst_fmt;
    int err = format_to_zimg(&src_fmt, src->pix_fmt, src->bpc,
                             src->w[0], src->h[0], &src->color);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: source picture cannot "
                 "be converted\n");
        return err;
    }
    err = format_to_zimg(&dst_fmt, target->pix_fmt, target->bpc,
                         dst_w, dst_h, &target->color);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: target picture cannot "
                 "be converted\n");
        return err;
    }

    VmafPictureConvertContext *c = malloc(sizeof(*c));
    if (!c) return -ENOMEM;
    memset(c, 0, sizeof(*c));
    c->target = *target;
    c->target.w = dst_w;
    c->target.h = dst_h;
    c->src.pix_fmt = src->pix_fmt;
    c->src.bpc = src->bpc;
    c->src.w = src->w[0];
    c->src.h = src->h[0];
    c->src.color = src->color;

    zimg_graph_builder_params params;
    const zimg_graph_builder_params *params_ptr = &params;
    zimg_graph_builder_params_default(&params, ZIMG_API_VERSION);
    /* Match FFmpeg's zscale (agamma=1): exact transfer functions are ~20x
     * slower for PQ and change scores negligibly. */
    params.allow_approximate_gamma = 1;
    if (target->resample_filter != VMAF_RESAMPLE_DEFAULT)
        params.resample_filter = resample_filter_to_zimg(target->resample_filter);

    c->graph = zimg_filter_graph_build(&src_fmt, &dst_fmt, params_ptr);
    if (!c->graph) {
        char err_msg[256];
        zimg_get_last_error(err_msg, sizeof(err_msg));
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: zimg_filter_graph_build "
                 "failed: %s\n", err_msg);
        err = -EINVAL;
        goto fail;
    }

    size_t tmp_size;
    if (zimg_filter_graph_get_tmp_size(c->graph, &tmp_size)) {
        char err_msg[256];
        zimg_get_last_error(err_msg, sizeof(err_msg));
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert_context_init: "
                 "zimg_filter_graph_get_tmp_size failed: %s\n", err_msg);
        err = -EINVAL;
        goto fail;
    }

    if (tmp_size) {
        c->tmp = aligned_malloc(tmp_size, 64);
        if (!c->tmp) {
            err = -ENOMEM;
            goto fail;
        }
    }

    *ctx = c;
    return 0;

fail:
    if (c->graph) zimg_filter_graph_free(c->graph);
    free(c);
    return err;
}

int vmaf_picture_convert(VmafPictureConvertContext *ctx, VmafPicture *dst,
                         const VmafPicture *src)
{
    if (!ctx || !dst || !src) return -EINVAL;

    if (src->pix_fmt != ctx->src.pix_fmt || src->bpc != ctx->src.bpc ||
        src->w[0] != ctx->src.w || src->h[0] != ctx->src.h ||
        !color_equal(&src->color, &ctx->src.color))
    {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert: source picture does not match the "
                 "format the context was initialized with\n");
        return -EINVAL;
    }

    int err = vmaf_picture_alloc(dst, ctx->target.pix_fmt, ctx->target.bpc,
                                 ctx->target.w, ctx->target.h);
    if (err) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert: could not allocate target picture "
                 "(pix_fmt %d, bpc %u, %ux%u)\n", (int) ctx->target.pix_fmt,
                 ctx->target.bpc, ctx->target.w, ctx->target.h);
        return err;
    }
    dst->color = ctx->target.color;

    zimg_image_buffer_const src_buf;
    zimg_image_buffer dst_buf;
    image_buffer_const_from_picture(&src_buf, src);
    image_buffer_from_picture(&dst_buf, dst);

    if (zimg_filter_graph_process(ctx->graph, &src_buf, &dst_buf, ctx->tmp,
                                  NULL, NULL, NULL, NULL))
    {
        char err_msg[256];
        zimg_get_last_error(err_msg, sizeof(err_msg));
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vmaf_picture_convert: zimg_filter_graph_process "
                 "failed: %s\n", err_msg);
        vmaf_picture_unref(dst);
        return -EINVAL;
    }

    return 0;
}

int vmaf_picture_convert_context_close(VmafPictureConvertContext *ctx)
{
    if (!ctx) return -EINVAL;
    zimg_filter_graph_free(ctx->graph);
    aligned_free(ctx->tmp);
    free(ctx);
    return 0;
}

#else /* !HAVE_ZIMG */

int vmaf_picture_convert_context_init(VmafPictureConvertContext **ctx,
                                      const VmafPicture *src,
                                      const VmafPictureConvertTarget *target)
{
    (void) ctx;
    (void) src;
    (void) target;
    vmaf_log(VMAF_LOG_LEVEL_ERROR,
             "vmaf_picture_convert_context_init: libvmaf was built without "
             "zimg support (configure with -Denable_zimg=true)\n");
    return -ENOTSUP;
}

int vmaf_picture_convert(VmafPictureConvertContext *ctx, VmafPicture *dst,
                         const VmafPicture *src)
{
    (void) ctx;
    (void) dst;
    (void) src;
    vmaf_log(VMAF_LOG_LEVEL_ERROR,
             "vmaf_picture_convert: libvmaf was built without zimg support "
             "(configure with -Denable_zimg=true)\n");
    return -ENOTSUP;
}

int vmaf_picture_convert_context_close(VmafPictureConvertContext *ctx)
{
    (void) ctx;
    vmaf_log(VMAF_LOG_LEVEL_ERROR,
             "vmaf_picture_convert_context_close: libvmaf was built without "
             "zimg support (configure with -Denable_zimg=true)\n");
    return -ENOTSUP;
}

#endif /* HAVE_ZIMG */
