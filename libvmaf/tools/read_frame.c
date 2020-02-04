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

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "file_io.h"
#include "read_frame.h"

/**
 * Note: stride is in terms of bytes
 */
static int read_image(FILE *rfile, void *buf, int width, int height, int stride, int elem_size)
{
	char *byte_ptr = buf;
	int i;
	int ret = 1;

	if (width <= 0 || height <= 0 || elem_size <= 0)
	{
		goto fail_or_end;
	}

	for (i = 0; i < height; ++i)
	{
		if (fread(byte_ptr, elem_size, width, rfile) != (size_t)width)
		{
			goto fail_or_end;
		}

		byte_ptr += stride;
	}

	ret = 0;

fail_or_end:
	return ret;
}

/**
 * Note: stride is in terms of bytes
 */
static int read_image_b(FILE * rfile, float *buf, float off, int width, int height, int stride)
{
    char *byte_ptr = (char *)buf;
	unsigned char *tmp_buf = 0;
	int i, j;
	int ret = 1;

	if (width <= 0 || height <= 0)
	{
		goto fail_or_end;
	}

	if (!(tmp_buf = malloc(width)))
	{
		goto fail_or_end;
	}

	for (i = 0; i < height; ++i)
	{
		float *row_ptr = (float *)byte_ptr;

		if (fread(tmp_buf, 1, width, rfile) != (size_t)width)
		{
			goto fail_or_end;
		}

		for (j = 0; j < width; ++j)
		{
			row_ptr[j] = tmp_buf[j] + off;
		}

		byte_ptr += stride;
	}
	ret = 0;

fail_or_end:
	free(tmp_buf);
	return ret;
}

/**
 * Note: stride is in terms of bytes; image is 10-bit little-endian
 */
static int read_image_w(FILE * rfile, float *buf, float off, int width, int height, int stride)
{
	// make sure unsigned short is 2 bytes
	assert(sizeof(unsigned short) == 2);

	char *byte_ptr = (char *)buf;
	unsigned short *tmp_buf = 0;
	int i, j;
	int ret = 1;

	if (width <= 0 || height <= 0)
	{
		goto fail_or_end;
	}

	if (!(tmp_buf = malloc(width * 2))) // '*2' to accommodate words
	{
		goto fail_or_end;
	}

	for (i = 0; i < height; ++i)
	{
		float *row_ptr = (float *)byte_ptr;

		if (fread(tmp_buf, 2, width, rfile) != (size_t)width) // '2' for word
		{
			goto fail_or_end;
		}

		for (j = 0; j < width; ++j)
		{
			row_ptr[j] = tmp_buf[j] / 4.0 + off; // '/4' to convert from 10 to 8-bit
		}

		byte_ptr += stride;
	}

	ret = 0;

fail_or_end:
	free(tmp_buf);
	return ret;
}

static int completed_frames = 0;

int read_frame(float *ref_data, float *dis_data, float *temp_data, int stride_byte, void *s)
{
    struct data *user_data = (struct data *)s;
    char *fmt = user_data->format;
    int w = user_data->width;
    int h = user_data->height;
    int ret;

    // read ref y
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
    {
        ret = read_image_b(user_data->ref_rfile, ref_data, 0, w, h, stride_byte);
    }
    else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
    {
        ret = read_image_w(user_data->ref_rfile, ref_data, 0, w, h, stride_byte);
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", fmt);
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
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
    {
        ret = read_image_b(user_data->dis_rfile, dis_data, 0, w, h, stride_byte);
    }
    else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
    {
        ret = read_image_w(user_data->dis_rfile, dis_data, 0, w, h, stride_byte);
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", fmt);
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
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
    {
        if (fread(temp_data, 1, user_data->offset, user_data->ref_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "ref fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
    {
        if (fread(temp_data, 2, user_data->offset, user_data->ref_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "ref fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", fmt);
        goto fail_or_end;
    }

    // dis skip u and v
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
    {
        if (fread(temp_data, 1, user_data->offset, user_data->dis_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "dis fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
    {
        if (fread(temp_data, 2, user_data->offset, user_data->dis_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "dis fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", fmt);
        goto fail_or_end;
    }

    fprintf(stderr, "Frame: %d/%d\r", completed_frames++, user_data->num_frames);


fail_or_end:
    return ret;
}

int read_noref_frame(float *dis_data, float *temp_data, int stride_byte, void *s, int offset)
{
    struct noref_data *user_data = (struct noref_data *)s;
    char *fmt = user_data->format;
    int w = user_data->width;
    int h = user_data->height;
    int ret;
    
    // if we have given a valid (non-negative) offset value, seek to that frame. This if statement
    // should only be entered from the second pass in motion.c for comparing motion between all frames.
    if (offset >= 0){
        fseek(user_data->dis_rfile, offset, SEEK_SET);
    }
    // read dis y
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
    {
        ret = read_image_b(user_data->dis_rfile, dis_data, 0, w, h, stride_byte);
    }
    else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
    {
        ret = read_image_w(user_data->dis_rfile, dis_data, 0, w, h, stride_byte);
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", fmt);
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
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
    {
        if (fread(temp_data, 1, user_data->offset, user_data->dis_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "dis fread u and v failed.\n");
            goto fail_or_end;
        }
        
    }
    else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
    {
        if (fread(temp_data, 2, user_data->offset, user_data->dis_rfile) != (size_t)user_data->offset)
        {
            fprintf(stderr, "dis fread u and v failed.\n");
            goto fail_or_end;
        }
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", fmt);
        goto fail_or_end;
    }


fail_or_end:
    return ret;
}

int get_frame_offset(const char *fmt, int w, int h, size_t *offset)
{
    if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv420p10le"))
    {
        if ((w * h) % 2 != 0)
        {
            fprintf(stderr, "(width * height) %% 2 != 0, width = %d, height = %d.\n", w, h);
            return 1;
        }
        *offset = w * h / 2;
    }
    else if (!strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv422p10le"))
    {
        *offset = w * h;
    }
    else if (!strcmp(fmt, "yuv444p") || !strcmp(fmt, "yuv444p10le"))
    {
        *offset = w * h * 2;
    }
    else
    {
        fprintf(stderr, "unknown format %s.\n", fmt);
        return 1;
    }
    return 0;
}
