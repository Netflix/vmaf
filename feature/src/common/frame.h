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

#ifndef FRAME_H_
#define FRAME_H_

#include <string.h>
#include "alloc.h"
#include "libvmaf.h"

struct data
{
    enum VmafPixelFormat format; /* yuv420p, yuv422p, yuv444p, yuv420p10le, yuv422p10le, yuv444p10le */
    int width;
    int height;
    size_t offset;
    FILE *ref_rfile;
    FILE *dis_rfile;
    int num_frames;
    int use_color;
};

struct noref_data
{
    enum VmafPixelFormat format; /* yuv420p, yuv422p, yuv444p, yuv420p10le, yuv422p10le, yuv444p10le */
    int width;
    int height;
    size_t offset;
    FILE *dis_rfile;
};

int read_frame(float *ref_data, float *dis_data, float *temp_data, int stride_byte, void *s);

int read_vmaf_picture(VmafPicture *ref_vmaf_pict, VmafPicture *dis_vmaf_pict, float *temp_data, void *s);

int read_noref_frame(float *dis_data, float *temp_data, int stride_byte, void *s);

int get_frame_offset(enum VmafPixelFormat fmt, int w, int h, size_t *offset);

int get_color_resolution(enum VmafPixelFormat fmt, unsigned int w, unsigned int h, unsigned int *w_u, unsigned int *h_u, unsigned int *w_v, unsigned int *h_v);

int get_stride_byte_from_width(unsigned int w);

enum VmafPixelFormat get_pix_fmt_from_input_char_ptr(const char *pix_fmt_option);

#endif /* FRAME_H_ */
