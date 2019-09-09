/**
 *
 *  Copyright 2016-2019 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "alloc.h"
#include "file_io.h"
#include "frame.h"

#define read_image_b       read_image_b2s
#define read_image_w       read_image_w2s

static int completed_frames = 0;

enum VmafPixelFormat get_pix_fmt_from_input_char_ptr(const char *pix_fmt_option)
{
    enum VmafPixelFormat pix_fmt;
    if (strcmp(pix_fmt_option, "yuv420p") == 0)
    {
        pix_fmt = VMAF_PIX_FMT_YUV420P;
    }
    else if (strcmp(pix_fmt_option, "yuv422p") == 0)
    {
        pix_fmt = VMAF_PIX_FMT_YUV422P;
    }
    else if (strcmp(pix_fmt_option, "yuv444p") == 0)
    {
        pix_fmt = VMAF_PIX_FMT_YUV444P;
    }
    else if (strcmp(pix_fmt_option, "yuv420p10le") == 0)
    {
        pix_fmt = VMAF_PIX_FMT_YUV420P10LE;
    }
    else if (strcmp(pix_fmt_option, "yuv422p10le") == 0)
    {
        pix_fmt = VMAF_PIX_FMT_YUV422P10LE;
    }
    else if (strcmp(pix_fmt_option, "yuv444p10le") == 0)
    {
        pix_fmt = VMAF_PIX_FMT_YUV444P10LE;
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", pix_fmt_option);
        pix_fmt = VMAF_PIX_FMT_UNKNOWN;
    }
    return pix_fmt;
}

const char* get_fmt_str_from_fmt_enum(enum VmafPixelFormat fmt_enum)
{
    switch (fmt_enum)
    {
        case VMAF_PIX_FMT_YUV420P: return "yuv420p";
        case VMAF_PIX_FMT_YUV422P: return "yuv422p";
        case VMAF_PIX_FMT_YUV444P: return "yuv444p";
        case VMAF_PIX_FMT_YUV420P10LE: return "yuv420p10le";
        case VMAF_PIX_FMT_YUV422P10LE: return "yuv422p10le";
        case VMAF_PIX_FMT_YUV444P10LE: return "yuv444p10le";
        default: return "unknown";
    }
}

int read_vmaf_picture(VmafPicture *ref_vmaf_pict, VmafPicture *dis_vmaf_pict, float *temp_data, void *s)
{
    struct data *user_data = (struct data *)s;
    enum VmafPixelFormat format = user_data->format;
    unsigned int w = user_data->width;
    unsigned int h = user_data->height;
    int ret;

    ref_vmaf_pict->pix_fmt = format;
    ref_vmaf_pict->w[0] = w;
    ref_vmaf_pict->h[0] = h;

    dis_vmaf_pict->pix_fmt = format;
    dis_vmaf_pict->w[0] = w;
    dis_vmaf_pict->h[0] = h;

    int chroma_resolution_ret = get_chroma_resolution(ref_vmaf_pict->pix_fmt, ref_vmaf_pict->w[0],
        ref_vmaf_pict->h[0], &(ref_vmaf_pict->w[1]), &(ref_vmaf_pict->h[1]),
        &(ref_vmaf_pict->w[2]), &(ref_vmaf_pict->h[2]));

    if (chroma_resolution_ret) {
        fprintf(stderr, "Calculating resolutions for chroma channels for ref failed.\n");
        return 1;
    }
    chroma_resolution_ret = get_chroma_resolution(dis_vmaf_pict->pix_fmt, dis_vmaf_pict->w[0],
        dis_vmaf_pict->h[0], &(dis_vmaf_pict->w[1]), &(dis_vmaf_pict->h[1]),
        &(dis_vmaf_pict->w[2]), &(dis_vmaf_pict->h[2]));

    if (chroma_resolution_ret) {
        fprintf(stderr, "Calculating resolutions for chroma channels for dis failed.\n");
        return 1;
    }

    ref_vmaf_pict->stride_byte[0] = get_stride_byte_from_width(ref_vmaf_pict->w[0]);
    ref_vmaf_pict->stride_byte[1] = get_stride_byte_from_width(ref_vmaf_pict->w[1]);
    ref_vmaf_pict->stride_byte[2] = get_stride_byte_from_width(ref_vmaf_pict->w[2]);

    dis_vmaf_pict->stride_byte[0] = get_stride_byte_from_width(dis_vmaf_pict->w[0]);
    dis_vmaf_pict->stride_byte[1] = get_stride_byte_from_width(dis_vmaf_pict->w[1]);
    dis_vmaf_pict->stride_byte[2] = get_stride_byte_from_width(dis_vmaf_pict->w[2]);

    enum VmafPixelFormat ref_fmt = ref_vmaf_pict->pix_fmt;
    enum VmafPixelFormat dis_fmt = dis_vmaf_pict->pix_fmt;

    // TODO: MORE CHECKS FOR RESOLUTION HERE

    // read ref y
    if (ref_fmt == VMAF_PIX_FMT_YUV420P || ref_fmt == VMAF_PIX_FMT_YUV422P || ref_fmt == VMAF_PIX_FMT_YUV444P)
    {
        ret = read_image_b(user_data->ref_rfile, ref_vmaf_pict->data[0], 0, ref_vmaf_pict->w[0], ref_vmaf_pict->h[0], ref_vmaf_pict->stride_byte[0]);
    }
    else if (ref_fmt == VMAF_PIX_FMT_YUV420P10LE || ref_fmt == VMAF_PIX_FMT_YUV422P10LE || ref_fmt == VMAF_PIX_FMT_YUV444P10LE)
    {
        ret = read_image_w(user_data->ref_rfile, ref_vmaf_pict->data[0], 0, ref_vmaf_pict->w[0], ref_vmaf_pict->h[0], ref_vmaf_pict->stride_byte[0]);
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(ref_fmt));
        return 1;
    }
    if (ret)
    {
        if (feof(user_data->ref_rfile))
        {
            ret = 2; // OK if end of file
        }
        return ret;
    }

    // read dis y
    if (dis_fmt == VMAF_PIX_FMT_YUV420P || dis_fmt == VMAF_PIX_FMT_YUV422P || dis_fmt == VMAF_PIX_FMT_YUV444P)
    {
        ret = read_image_b(user_data->dis_rfile, dis_vmaf_pict->data[0], 0, dis_vmaf_pict->w[0], dis_vmaf_pict->h[0], dis_vmaf_pict->stride_byte[0]);
    }
    else if (dis_fmt == VMAF_PIX_FMT_YUV420P10LE || dis_fmt == VMAF_PIX_FMT_YUV422P10LE || dis_fmt == VMAF_PIX_FMT_YUV444P10LE)
    {
        ret = read_image_w(user_data->dis_rfile, dis_vmaf_pict->data[0], 0, dis_vmaf_pict->w[0], dis_vmaf_pict->h[0], dis_vmaf_pict->stride_byte[0]);
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(dis_fmt));
        return 1;
    }
    if (ret)
    {
        if (feof(user_data->dis_rfile))
        {
            ret = 2; // OK if end of file
        }
        return ret;
    }

    // read ref u
    if (ref_fmt == VMAF_PIX_FMT_YUV420P || ref_fmt == VMAF_PIX_FMT_YUV422P || ref_fmt == VMAF_PIX_FMT_YUV444P)
    {
        ret = read_image_b(user_data->ref_rfile, ref_vmaf_pict->data[1], 0, ref_vmaf_pict->w[1], ref_vmaf_pict->h[1], ref_vmaf_pict->stride_byte[1]);
    }
    else if (ref_fmt == VMAF_PIX_FMT_YUV420P10LE || ref_fmt == VMAF_PIX_FMT_YUV422P10LE || ref_fmt == VMAF_PIX_FMT_YUV444P10LE)
    {
        ret = read_image_w(user_data->ref_rfile, ref_vmaf_pict->data[1], 0, ref_vmaf_pict->w[1], ref_vmaf_pict->h[1], ref_vmaf_pict->stride_byte[1]);
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(ref_fmt));
        return 1;
    }
    if (ret)
    {
        if (feof(user_data->ref_rfile))
        {
            ret = 2; // OK if end of file
        }
        return ret;
    }

    // read dis u
    if (dis_fmt == VMAF_PIX_FMT_YUV420P || dis_fmt == VMAF_PIX_FMT_YUV422P || dis_fmt == VMAF_PIX_FMT_YUV444P)
    {
        ret = read_image_b(user_data->dis_rfile, dis_vmaf_pict->data[1], 0, dis_vmaf_pict->w[1], dis_vmaf_pict->h[1], dis_vmaf_pict->stride_byte[1]);
    }
    else if (dis_fmt == VMAF_PIX_FMT_YUV420P10LE || dis_fmt == VMAF_PIX_FMT_YUV422P10LE || dis_fmt == VMAF_PIX_FMT_YUV444P10LE)
    {
        ret = read_image_w(user_data->dis_rfile, dis_vmaf_pict->data[1], 0, dis_vmaf_pict->w[1], dis_vmaf_pict->h[1], dis_vmaf_pict->stride_byte[1]);
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(dis_fmt));
        return 1;
    }
    if (ret)
    {
        if (feof(user_data->dis_rfile))
        {
            ret = 2; // OK if end of file
        }
        return ret;
    }

    // read ref v
    if (ref_fmt == VMAF_PIX_FMT_YUV420P || ref_fmt == VMAF_PIX_FMT_YUV422P || ref_fmt == VMAF_PIX_FMT_YUV444P)
    {
        ret = read_image_b(user_data->ref_rfile, ref_vmaf_pict->data[2], 0, ref_vmaf_pict->w[2], ref_vmaf_pict->h[2], ref_vmaf_pict->stride_byte[2]);
    }
    else if (ref_fmt == VMAF_PIX_FMT_YUV420P10LE || ref_fmt == VMAF_PIX_FMT_YUV422P10LE || ref_fmt == VMAF_PIX_FMT_YUV444P10LE)
    {
        ret = read_image_w(user_data->ref_rfile, ref_vmaf_pict->data[2], 0, ref_vmaf_pict->w[2], ref_vmaf_pict->h[2], ref_vmaf_pict->stride_byte[2]);
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(ref_fmt));
        return 1;
    }
    if (ret)
    {
        if (feof(user_data->ref_rfile))
        {
            ret = 2; // OK if end of file
        }
        return ret;
    }

    // read dis v
    if (dis_fmt == VMAF_PIX_FMT_YUV420P || dis_fmt == VMAF_PIX_FMT_YUV422P || dis_fmt == VMAF_PIX_FMT_YUV444P)
    {
        ret = read_image_b(user_data->dis_rfile, dis_vmaf_pict->data[2], 0, dis_vmaf_pict->w[2], dis_vmaf_pict->h[2], dis_vmaf_pict->stride_byte[2]);
    }
    else if (dis_fmt == VMAF_PIX_FMT_YUV420P10LE || dis_fmt == VMAF_PIX_FMT_YUV422P10LE || dis_fmt == VMAF_PIX_FMT_YUV444P10LE)
    {
        ret = read_image_w(user_data->dis_rfile, dis_vmaf_pict->data[2], 0, dis_vmaf_pict->w[2], dis_vmaf_pict->h[2], dis_vmaf_pict->stride_byte[2]);
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(dis_fmt));
        return 1;
    }
    if (ret)
    {
        if (feof(user_data->dis_rfile))
        {
            ret = 2; // OK if end of file
        }
        return ret;
    }

fail_or_end:
    return ret;
}

int read_frame(float *ref_data, float *dis_data, float *temp_data, int stride_byte, void *s)
{
    struct data *user_data = (struct data *)s;
    enum VmafPixelFormat fmt_enum = user_data->format;
    int w = user_data->width;
    int h = user_data->height;
    int ret;

    // read ref y
    if (fmt_enum == VMAF_PIX_FMT_YUV420P || fmt_enum == VMAF_PIX_FMT_YUV422P || fmt_enum == VMAF_PIX_FMT_YUV444P)
    {
        ret = read_image_b(user_data->ref_rfile, ref_data, 0, w, h, stride_byte);
    }
    else if (fmt_enum == VMAF_PIX_FMT_YUV420P10LE || fmt_enum == VMAF_PIX_FMT_YUV422P10LE || fmt_enum == VMAF_PIX_FMT_YUV444P10LE)
    {
        ret = read_image_w(user_data->ref_rfile, ref_data, 0, w, h, stride_byte);
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", get_fmt_str_from_fmt_enum(fmt_enum));
        return 1;
    }
    if (ret)
    {
        if (feof(user_data->ref_rfile))
        {
            ret = 2; // OK if end of file
        }
        return ret;
    }

    // read dis y
    if (fmt_enum == VMAF_PIX_FMT_YUV420P || fmt_enum == VMAF_PIX_FMT_YUV422P || fmt_enum == VMAF_PIX_FMT_YUV444P)
    {
        ret = read_image_b(user_data->dis_rfile, dis_data, 0, w, h, stride_byte);
    }
    else if (fmt_enum == VMAF_PIX_FMT_YUV420P10LE || fmt_enum == VMAF_PIX_FMT_YUV422P10LE || fmt_enum == VMAF_PIX_FMT_YUV444P10LE)
    {
        ret = read_image_w(user_data->dis_rfile, dis_data, 0, w, h, stride_byte);
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(fmt_enum));
        return 1;
    }
    if (ret)
    {
        if (feof(user_data->dis_rfile))
        {
            ret = 2; // OK if end of file
        }
        return ret;
    }

    // ref skip u and v
    if (fmt_enum == VMAF_PIX_FMT_YUV420P || fmt_enum == VMAF_PIX_FMT_YUV422P || fmt_enum == VMAF_PIX_FMT_YUV444P)
    {
        if (fread(temp_data, 1, user_data->offset, user_data->ref_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "ref fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else if (fmt_enum == VMAF_PIX_FMT_YUV420P10LE || fmt_enum == VMAF_PIX_FMT_YUV422P10LE || fmt_enum == VMAF_PIX_FMT_YUV444P10LE)
    {
        if (fread(temp_data, 2, user_data->offset, user_data->ref_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "ref fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(fmt_enum));
        goto fail_or_end;
    }

    // dis skip u and v
    if (fmt_enum == VMAF_PIX_FMT_YUV420P || fmt_enum == VMAF_PIX_FMT_YUV422P || fmt_enum == VMAF_PIX_FMT_YUV444P)
    {
        if (fread(temp_data, 1, user_data->offset, user_data->dis_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "dis fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else if (fmt_enum == VMAF_PIX_FMT_YUV420P10LE || fmt_enum == VMAF_PIX_FMT_YUV422P10LE || fmt_enum == VMAF_PIX_FMT_YUV444P10LE)
    {
        if (fread(temp_data, 2, user_data->offset, user_data->dis_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "dis fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(fmt_enum));
        goto fail_or_end;
    }
    
    fprintf(stderr, "Frame: %d/%d\r", completed_frames++, user_data->num_frames);

fail_or_end:
    return ret;
}

int read_noref_frame(float *dis_data, float *temp_data, int stride_byte, void *s)
{
    struct noref_data *user_data = (struct noref_data *)s;
    enum VmafPixelFormat fmt_enum = user_data->format;
    int w = user_data->width;
    int h = user_data->height;
    int ret;

    // read dis y
    if (fmt_enum == VMAF_PIX_FMT_YUV420P || fmt_enum == VMAF_PIX_FMT_YUV422P || fmt_enum == VMAF_PIX_FMT_YUV444P)
    {
        ret = read_image_b(user_data->dis_rfile, dis_data, 0, w, h, stride_byte);
    }
    else if (fmt_enum == VMAF_PIX_FMT_YUV420P10LE || fmt_enum == VMAF_PIX_FMT_YUV422P10LE || fmt_enum == VMAF_PIX_FMT_YUV444P10LE)
    {
        ret = read_image_w(user_data->dis_rfile, dis_data, 0, w, h, stride_byte);
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(fmt_enum));
        return 1;
    }
    if (ret)
    {
        if (feof(user_data->dis_rfile))
        {
            ret = 2; // OK if end of file
        }
        return ret;
    }

    // dis skip u and v
    if (fmt_enum == VMAF_PIX_FMT_YUV420P || fmt_enum == VMAF_PIX_FMT_YUV422P || fmt_enum == VMAF_PIX_FMT_YUV444P)
    {
        if (fread(temp_data, 1, user_data->offset, user_data->dis_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "dis fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else if (fmt_enum == VMAF_PIX_FMT_YUV420P10LE || fmt_enum == VMAF_PIX_FMT_YUV422P10LE || fmt_enum == VMAF_PIX_FMT_YUV444P10LE)
    {
        if (fread(temp_data, 2, user_data->offset, user_data->dis_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "dis fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else
    {
        fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(fmt_enum));
        goto fail_or_end;
    }

fail_or_end:
    return ret;
}

int get_frame_offset(enum VmafPixelFormat fmt_enum, int w, int h, size_t *offset)
{
    switch (fmt_enum)
    {
        case VMAF_PIX_FMT_YUV420P:
        case VMAF_PIX_FMT_YUV420P10LE:

            if ((w * h) % 2 != 0)
            {
                fprintf(stderr, "(width * height) %% 2 != 0, width = %d, height = %d.\n", w, h);
                return 1;
            }
            *offset = w * h / 2;
            break;

        case VMAF_PIX_FMT_YUV422P:
        case VMAF_PIX_FMT_YUV422P10LE:

            *offset = w * h;
            break;

        case VMAF_PIX_FMT_YUV444P:
        case VMAF_PIX_FMT_YUV444P10LE:

            *offset = w * h * 2;
            break;

        default:

            fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(fmt_enum));
            return 1;
    }
    return 0;
}

int get_chroma_resolution(enum VmafPixelFormat fmt_enum, unsigned int w, unsigned int h, unsigned int *w_u, unsigned int *h_u, unsigned int *w_v, unsigned int *h_v) {

    // check that both width and height are positive
    if ((w <= 0) || (h <= 0))
    {
        fprintf(stderr, "Width or height is not positive, width = %d, height = %d.\n", w, h);
        return 1;
    }

    switch (fmt_enum)
    {
        case VMAF_PIX_FMT_YUV420P:
        case VMAF_PIX_FMT_YUV420P10LE:
            // check that both width and height are even, i.e., their product is divisible by 2
            if ((w * h) % 2 != 0)
            {
                fprintf(stderr, "(width * height) %% 2 != 0, width = %d, height = %d.\n", w, h);
                return 1;
            }
            *w_u = w / 2;
            *w_v = w / 2;
            *h_u = h / 2;
            *h_v = h / 2;
            break;

        case VMAF_PIX_FMT_YUV422P:
        case VMAF_PIX_FMT_YUV422P10LE:

            // need to double check this
            *w_u = w / 2;
            *w_v = w / 2;
            *h_u = h;
            *h_v = h;
            break;

        case VMAF_PIX_FMT_YUV444P:
        case VMAF_PIX_FMT_YUV444P10LE:

            *w_u = w;
            *w_v = w;
            *h_u = h;
            *h_v = h;
            break;

        default:
            fprintf(stderr, "Unknown format %s.\n", get_fmt_str_from_fmt_enum(fmt_enum));
            return 1;
    }

    if (*w_u <= 0 || *h_u <= 0 || (size_t)*w_u > ALIGN_FLOOR(INT_MAX) / sizeof(float))
    {
        fprintf(stderr, "Invalid width and height for u, width = %d, height = %d.\n", *w_u, *h_u);
        return 1;
    }

    if (*w_v <= 0 || *h_v <= 0 || (size_t)*w_v > ALIGN_FLOOR(INT_MAX) / sizeof(float))
    {
        fprintf(stderr, "Invalid width and height for v, width = %d, height = %d.\n", *w_v, *h_v);
        return 1;
    }

    return 0;
}

int get_stride_byte_from_width(unsigned int w) {
    return ALIGN_CEIL(w * sizeof(float));
}
