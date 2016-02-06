/**
 *
 *  Copyright 2016 Netflix, Inc.
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
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "all_options.h"
#include "common/alloc.h"
#include "common/file_io.h"
#include "main.h"

#ifdef ALL_OPT_SINGLE_PRECISION
	typedef float number_t;

	#define read_image_b       read_image_b2s
	#define read_image_w       read_image_w2s
	int compute_adm(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score);
	int compute_ansnr(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score);
	int compute_vif(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score);

#else
	typedef double number_t;

	#define read_image_b       read_image_b2d
	#define read_image_w       read_image_w2d
	int compute_adm(const double *ref, const double *dis, int w, int h, int ref_stride, int dis_stride, double *score);
	int compute_ansnr(const double *ref, const double *dis, int w, int h, int ref_stride, int dis_stride, double *score);
	int compute_vif(const double *ref, const double *dis, int w, int h, int ref_stride, int dis_stride, double *score);

#endif



int all(const char *ref_path, const char *dis_path, int w, int h, const char *fmt)
{
	double score = 0;
	number_t *ref_buf = 0;
	number_t *dis_buf = 0;

	// prev_blur_buf, blur_buf, temp_buf for motion only
	number_t *prev_blur_buf = 0;
	number_t *blur_buf = 0;
	number_t *temp_buf = 0;

	FILE *ref_rfile = 0;
	FILE *dis_rfile = 0;
	size_t data_sz;
	int stride;
	int ret = 1;

	if (w <= 0 || h <= 0 || (size_t)w > ALIGN_FLOOR(INT_MAX) / sizeof(number_t))
	{
		goto fail_or_end;
	}

	stride = ALIGN_CEIL(w * sizeof(number_t));

	if ((size_t)h > SIZE_MAX / stride)
	{
		goto fail_or_end;
	}

	data_sz = (size_t)stride * h;

	if (!(ref_buf = aligned_malloc(data_sz, MAX_ALIGN)))
	{
		printf("error: aligned_malloc failed for ref_buf.\n");
		fflush(stdout);
		goto fail_or_end;
	}
	if (!(dis_buf = aligned_malloc(data_sz, MAX_ALIGN)))
	{
		printf("error: aligned_malloc failed for dis_buf.\n");
		fflush(stdout);
		goto fail_or_end;
	}

	// prev_blur_buf, blur_buf, temp_buf for motion only
	if (!(prev_blur_buf = aligned_malloc(data_sz, MAX_ALIGN)))
	{
		printf("error: aligned_malloc failed for prev_blur_buf.\n");
		fflush(stdout);
		goto fail_or_end;
	}
	if (!(blur_buf = aligned_malloc(data_sz, MAX_ALIGN)))
	{
		printf("error: aligned_malloc failed for blur_buf.\n");
		fflush(stdout);
		goto fail_or_end;
	}
	if (!(temp_buf = aligned_malloc(data_sz, MAX_ALIGN)))
	{
		printf("error: aligned_malloc failed for temp_buf.\n");
		fflush(stdout);
		goto fail_or_end;
	}

	if (!(ref_rfile = fopen(ref_path, "rb")))
	{
		printf("error: fopen ref_path %s failed.\n", ref_path);
		fflush(stdout);
		goto fail_or_end;
	}
	if (!(dis_rfile = fopen(dis_path, "rb")))
	{
		printf("error: fopen dis_path %s failed.\n", dis_path);
		fflush(stdout);
		goto fail_or_end;
	}

	size_t offset;
	if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv420p10le"))
	{
		if ((w * h) % 2 != 0)
		{
			printf("error: (w * h) %% 2 != 0, w = %d, h = %d.\n", w, h);
			fflush(stdout);
			goto fail_or_end;
		}
		offset = w * h / 2;
	}
	else if (!strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv422p10le"))
	{
		offset = w * h;
	}
	else if (!strcmp(fmt, "yuv444p") || !strcmp(fmt, "yuv444p10le"))
	{
		offset = w * h * 2;
	}
	else
	{
		printf("error: unknown format %s.\n", fmt);
		fflush(stdout);
		goto fail_or_end;
	}

	int frm_idx = 0;
	while ((!feof(ref_rfile)) && (!feof(dis_rfile)))
	{
		// read ref y
		if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
		{
			ret = read_image_b(ref_rfile, ref_buf, 0, w, h, stride);
		}
		else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
		{
			ret = read_image_w(ref_rfile, ref_buf, 0, w, h, stride);
		}
		else
		{
			printf("error: unknown format %s.\n", fmt);
			fflush(stdout);
			goto fail_or_end;
		}
		if (ret)
		{
			goto fail_or_end;
		}

		// read dis y
		if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
		{
			ret = read_image_b(dis_rfile, dis_buf, 0, w, h, stride);
		}
		else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
		{
			ret = read_image_w(dis_rfile, dis_buf, 0, w, h, stride);
		}
		else
		{
			printf("error: unknown format %s.\n", fmt);
			fflush(stdout);
			goto fail_or_end;
		}
		if (ret)
		{
			goto fail_or_end;
		}

		// compute & print for adm, ansnr and vif
		if ((ret = compute_adm(ref_buf, dis_buf, w, h, stride, stride, &score)))
		{
			printf("error: compute_adm failed.\n");
			fflush(stdout);
			goto fail_or_end;
		}
		printf("adm: %d %f\n", frm_idx, score);
		fflush(stdout);

		if ((ret = compute_ansnr(ref_buf, dis_buf, w, h, stride, stride, &score)))
		{
			printf("error: compute_ansnr failed.\n");
			fflush(stdout);
			goto fail_or_end;
		}
		printf("ansnr: %d %f\n", frm_idx, score);
		fflush(stdout);

		if ((ret = compute_vif(ref_buf, dis_buf, w, h, stride, stride, &score)))
		{
			printf("error: compute_vif failed.\n");
			fflush(stdout);
			goto fail_or_end;
		}
		printf("vif: %d %f\n", frm_idx, score);
		fflush(stdout);

		// ref skip u and v
		if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
		{
			ret = fseek(ref_rfile, offset, SEEK_CUR);
		}
		else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
		{
			ret = fseek(ref_rfile, offset * 2, SEEK_CUR);
		}
		else
		{
			printf("error: unknown format %s.\n", fmt);
			fflush(stdout);
			goto fail_or_end;
		}
		if (ret)
		{
			printf("error: fseek failed.\n");
			fflush(stdout);
			goto fail_or_end;
		}

		// dis skip u and v
		if (!strcmp(fmt, "yuv420p") || !strcmp(fmt, "yuv422p") || !strcmp(fmt, "yuv444p"))
		{
			ret = fseek(dis_rfile, offset, SEEK_CUR);
		}
		else if (!strcmp(fmt, "yuv420p10le") || !strcmp(fmt, "yuv422p10le") || !strcmp(fmt, "yuv444p10le"))
		{
			ret = fseek(dis_rfile, offset * 2, SEEK_CUR);
		}
		else
		{
			printf("error: unknown format %s.\n", fmt);
			fflush(stdout);
			goto fail_or_end;
		}
		if (ret)
		{
			printf("error: fseek failed.\n");
			fflush(stdout);
			goto fail_or_end;
		}

		frm_idx++;
	}

	ret = 0;

fail_or_end:
	if (ref_rfile)
	{
		fclose(ref_rfile);
	}
	if (dis_rfile)
	{
		fclose(dis_rfile);
	}
	aligned_free(ref_buf);
	aligned_free(dis_buf);

	// for motion
	aligned_free(prev_blur_buf);
	aligned_free(blur_buf);
	aligned_free(temp_buf);

	return ret;
}
