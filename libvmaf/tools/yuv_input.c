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

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#define yuv_fileno _fileno
#else
#include <unistd.h>
#define yuv_fileno fileno
#endif
#include <sys/stat.h>

#include "vidinput.h"

#include "libvmaf/picture.h"

/** Linkage will break without this if using a C++ compiler, and will issue
 * warnings without this for a C compiler*/
#if defined(__cplusplus)
#define OC_EXTERN extern
#else
#define OC_EXTERN
#endif

typedef struct yuv_input {
    FILE *fin;
    unsigned width, height;
    enum VmafPixelFormat pix_fmt;
    unsigned bitdepth;
    size_t dst_buf_sz;
    uint8_t *dst_buf;
    int src_c_dec_v, src_c_dec_h;
    int dst_c_dec_h, dst_c_dec_v;
} yuv_input;

/* Validate file size against declared geometry + bit depth so that a
 * mismatched --bitdepth flag surfaces a clear error instead of heap
 * corruption (malloc fastbin misalignment) at the first fread.
 *
 * Exits with code 2 on any geometry/depth mismatch (following the
 * cli_parse.c convention of calling exit() directly for bad-usage errors).
 * Returns silently when the fd is not a regular file (pipe, socket) — the
 * reader will discover EOF naturally.
 *
 * Uses fstat rather than fseek/ftell to avoid disturbing the stream
 * position and to correctly handle sizes >2 GiB via the
 * _LARGEFILE64_SOURCE / _FILE_OFFSET_BITS=64 definitions in vidinput.h.
 */
static void yuv_check_file_size(FILE *fin, const yuv_input *yuv)
{
    struct stat st;
    if (fstat(yuv_fileno(fin), &st) != 0 || !S_ISREG(st.st_mode))
        return; /* pipe or fstat failure — skip, let reader hit EOF */

    off_t file_sz = st.st_size;
    size_t frame_sz = yuv->dst_buf_sz;
    unsigned bpp = yuv->bitdepth > 8u ? 2u : 1u;
    const char *fmt_name = yuv->pix_fmt == VMAF_PIX_FMT_YUV420P ? "yuv420p" :
                           yuv->pix_fmt == VMAF_PIX_FMT_YUV422P ? "yuv422p" :
                                                                  "yuv444p";

    if (file_sz < (off_t)frame_sz) {
        (void)fprintf(stderr,
                      "yuv: file too small for declared geometry — "
                      "need at least %zu bytes for one %ux%u %u-bit %s frame, "
                      "got %lld bytes\n",
                      frame_sz, yuv->width, yuv->height, yuv->bitdepth, fmt_name,
                      (long long)file_sz);
        exit(
            2); // NOLINT(concurrency-mt-unsafe) — CLI single-threaded at open time; mirrors cli_parse.c exit() pattern
    }
    if (file_sz % (off_t)frame_sz != 0) {
        (void)fprintf(stderr,
                      "yuv: file size mismatch — expected a multiple of %zu bytes "
                      "for %ux%u %u-bit %s, got %lld bytes "
                      "(hint: check --bitdepth and --pixel_format; "
                      "%u-bit frames need %u byte%s per sample)\n",
                      frame_sz, yuv->width, yuv->height, yuv->bitdepth, fmt_name,
                      (long long)file_sz, yuv->bitdepth, bpp, bpp == 1u ? "" : "s");
        exit(
            2); // NOLINT(concurrency-mt-unsafe) — CLI single-threaded at open time; mirrors cli_parse.c exit() pattern
    }
}

static yuv_input *yuv_input_open(FILE *_fin, unsigned width, unsigned height,
                                 enum VmafPixelFormat pix_fmt, unsigned bitdepth)
{
    yuv_input *yuv = malloc(sizeof(*yuv));
    if (!yuv) {
        (void)fprintf(stderr, "Could not allocate yuv reader state.\n");
        return NULL;
    }

    yuv->fin = _fin;
    yuv->width = width;
    yuv->height = height;
    yuv->pix_fmt = pix_fmt;
    yuv->bitdepth = bitdepth;
    bool hbd = yuv->bitdepth > 8;

    switch (yuv->pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        yuv->src_c_dec_h = yuv->dst_c_dec_h = yuv->src_c_dec_v = yuv->dst_c_dec_v = 2;
        yuv->dst_buf_sz =
            (yuv->width * yuv->height + 2 * ((yuv->width + 1) / 2) * ((yuv->height + 1) / 2))
            << hbd;
        break;
    case VMAF_PIX_FMT_YUV422P:
        yuv->src_c_dec_h = yuv->dst_c_dec_h = 2;
        yuv->src_c_dec_v = yuv->dst_c_dec_v = 1;
        yuv->dst_buf_sz = (yuv->width * yuv->height + 2 * (((yuv->width + 1) / 2) * yuv->height))
                          << hbd;
        break;
    case VMAF_PIX_FMT_YUV444P:
        yuv->src_c_dec_h = yuv->dst_c_dec_h = yuv->src_c_dec_v = yuv->dst_c_dec_v = 1;
        yuv->dst_buf_sz = (yuv->width * yuv->height * 3) << hbd;
        break;
    default:
        goto fail;
    }

    yuv_check_file_size(_fin, yuv); /* exits with code 2 on mismatch */

    yuv->dst_buf = malloc(yuv->dst_buf_sz);
    if (!yuv->dst_buf) {
        (void)fprintf(stderr, "Could not allocate yuv reader buffer.\n");
        goto fail;
    }

    return yuv;

fail:
    free(yuv);
    return NULL;
}

static int pix_fmt_map(enum VmafPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        return PF_420;
    case VMAF_PIX_FMT_YUV422P:
        return PF_422;
    case VMAF_PIX_FMT_YUV444P:
        return PF_444;
    default:
        return 0;
    }
}

static void yuv_input_get_info(yuv_input *_yuv, video_input_info *_info)
{
    memset(_info, 0, sizeof(*_info));
    _info->frame_w = _info->pic_w = _yuv->width;
    _info->frame_h = _info->pic_h = _yuv->height;
    _info->pixel_fmt = pix_fmt_map(_yuv->pix_fmt);
    _info->depth = _yuv->bitdepth;
}

static int yuv_input_fetch_frame(yuv_input *yuv, FILE *fin, video_input_ycbcr _ycbcr,
                                 const char _tag[5])
{
    size_t bytes_read = fread(yuv->dst_buf, 1, yuv->dst_buf_sz, fin);
    if (bytes_read == 0)
        return 0;
    if (bytes_read != yuv->dst_buf_sz) {
        (void)fprintf(stderr, "Error reading YUV frame data.\n");
        return -1;
    }

    (void)_tag;

    unsigned xstride = (yuv->bitdepth > 8) ? 2 : 1;
    ptrdiff_t pic_sz = (ptrdiff_t)yuv->width * yuv->height * xstride;
    unsigned frame_c_w = yuv->width / yuv->dst_c_dec_h;
    unsigned frame_c_h = yuv->height / yuv->dst_c_dec_v;
    unsigned c_w = (yuv->width + yuv->dst_c_dec_h - 1) / yuv->dst_c_dec_h;
    unsigned c_h = (yuv->height + yuv->dst_c_dec_v - 1) / yuv->dst_c_dec_v;
    unsigned c_sz = c_w * c_h * xstride;

    _ycbcr[0].width = yuv->width;
    _ycbcr[0].height = yuv->height;
    _ycbcr[0].stride = yuv->width * xstride;
    _ycbcr[0].data = yuv->dst_buf;
    _ycbcr[1].width = frame_c_w;
    _ycbcr[1].height = frame_c_h;
    _ycbcr[1].stride = c_w * xstride;
    _ycbcr[1].data = yuv->dst_buf + pic_sz;
    _ycbcr[2].width = frame_c_w;
    _ycbcr[2].height = frame_c_h;
    _ycbcr[2].stride = c_w * xstride;
    _ycbcr[2].data = _ycbcr[1].data + c_sz;

    return 1;
}

static void yuv_input_close(yuv_input *_yuv)
{
    free(_yuv->dst_buf);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) — extern linkage required: vidinput.c references this symbol via `extern video_input_vtbl YUV_INPUT_VTBL`
OC_EXTERN const video_input_vtbl YUV_INPUT_VTBL = {
    (raw_input_open_func)yuv_input_open, (video_input_open_func)NULL,
    (video_input_get_info_func)yuv_input_get_info,
    (video_input_fetch_frame_func)yuv_input_fetch_frame, (video_input_close_func)yuv_input_close};
