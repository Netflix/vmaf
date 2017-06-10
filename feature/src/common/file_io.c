/**
 *
 *  Copyright 2016-2017 Netflix, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

/**
 * Note: stride is in terms of bytes
 */
int read_image(FILE *data, void *buf, int width, int height, int stride, int elem_size)
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
		if (fread(byte_ptr, elem_size, width, data) != (size_t)width)
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
int write_image(FILE *wfile, const void *buf, int width, int height, int stride, int elem_size)
{
	const char *byte_ptr = buf;
	int i;
	int ret = 1;

	if (width <= 0 || height <= 0 || elem_size <= 0)
	{
		goto fail_or_end;
	}

	for (i = 0; i < height; ++i)
	{
		if (fwrite(byte_ptr, elem_size, width, wfile) != (size_t)width)
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
int read_image_b2s(uint8_t *data, float *buf, float off, int width, int height, int stride)
{
	printf("success\n");
	uint8_t *ptr = data;
	char *byte_ptr = (char *)buf;
	unsigned char *tmp_buf = 0;
	int i, j;
	int ret = 1;
	printf("success1\n");

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
	printf("success2\n");
		float *row_ptr = (float *)byte_ptr;

		for (j = 0; j < width; ++j)
		{
		printf("success3-3\n");
		//printf("value = %u\n",ptr[j]);
			//row_ptr[j] = (float)ptr[j] + off;
		}

		byte_ptr += stride;
		ptr += stride;
	}

	ret = 0;

fail_or_end:
	free(tmp_buf);
	return ret;
}

/**
 * Note: stride is in terms of bytes
 */
int read_image_b2d(uint8_t *data, double *buf, double off, int width, int height, int stride)
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
		double *row_ptr = (double *)byte_ptr;

		if (fread(tmp_buf, 1, width, data) != (size_t)width)
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
int read_image_w2s(uint8_t *data, float *buf, float off, int width, int height, int stride)
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

		if (fread(tmp_buf, 2, width, data) != (size_t)width) // '2' for word
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

/**
 * Note: stride is in terms of bytes, image is 10-bit little endian
 */
int read_image_w2d(uint8_t *data, double *buf, double off, int width, int height, int stride)
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
		double *row_ptr = (double *)byte_ptr;

		if (fread(tmp_buf, 2, width, data) != (size_t)width) // '2' for word
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

/**
 * Note: stride is in terms of bytes
 */
int offset_image_s(float *buf, float off, int width, int height, int stride)
{
	char *byte_ptr = (char *)buf;
	int ret = 1;
	int i, j;

	for (i = 0; i < height; ++i)
	{
		float *row_ptr = (float *)byte_ptr;

		for (j = 0; j < width; ++j)
		{
			row_ptr[j] += off;
		}

		byte_ptr += stride;
	}

	ret = 0;

	return ret;
}

/**
 * Note: stride is in terms of bytes
 */
int offset_image_d(double *buf, double off, int width, int height, int stride)
{
	char *byte_ptr = (char *)buf;
	int ret = 1;
	int i, j;

	for (i = 0; i < height; ++i)
	{
		double *row_ptr = (double *)byte_ptr;

		for (j = 0; j < width; ++j)
		{
			row_ptr[j] += off;
		}

		byte_ptr += stride;
	}

	ret = 0;

	return ret;

}

