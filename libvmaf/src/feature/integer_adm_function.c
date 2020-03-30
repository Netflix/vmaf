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

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mem.h"
#include "adm_options.h"
#include "adm_tools.h"
#include "integer_adm_tools.h"
#include "offset.h"

typedef adm_dwt_band_t_s adm_dwt_band_t;

static char *integer_init_dwt_band(integer_adm_dwt_band_t_s *band, char *data_top, size_t buf_sz_one)
{
    band->band_a = (int16_t *)data_top; data_top += buf_sz_one;
    band->band_h = (int16_t *)data_top; data_top += buf_sz_one;
    band->band_v = (int16_t *)data_top; data_top += buf_sz_one;
    band->band_d = (int16_t *)data_top; data_top += buf_sz_one;
    return data_top;
}

static char *integer_i4_init_dwt_band(integer_i4_adm_dwt_band_t_s *band, char *data_top, size_t buf_sz_one)
{
	band->band_a = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_h = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_v = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_d = (int32_t *)data_top;
	data_top += buf_sz_one;
	return data_top;
}

static char *integer_init_dwt_band_hvd(integer_adm_dwt_band_t_s *band, char *data_top, size_t buf_sz_one)
{
	band->band_a = NULL;
	band->band_h = (int16_t *)data_top; data_top += buf_sz_one;
	band->band_v = (int16_t *)data_top; data_top += buf_sz_one;
	band->band_d = (int16_t *)data_top; data_top += buf_sz_one;
	return data_top;
}

static char *integer_i4_init_dwt_band_hvd(integer_i4_adm_dwt_band_t_s *band, char *data_top, size_t buf_sz_one)
{
	band->band_a = NULL;
	band->band_h = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_v = (int32_t *)data_top;
	data_top += buf_sz_one;
	band->band_d = (int32_t *)data_top;
	data_top += buf_sz_one;
	return data_top;
}

int integer_compute_adm(const int16_t *ref, const int16_t *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores, double border_factor, int inp_size_bits)
{
#ifdef ADM_OPT_SINGLE_PRECISION
	double numden_limit = 1e-2 * (w * h) / (1920.0 * 1080.0);
#else
	double numden_limit = 1e-10 * (w * h) / (1920.0 * 1080.0);
#endif
	float *data_buf = 0;
	char *data_top;

	char *ind_buf_y = 0, *buf_y_orig = 0;
	char *ind_buf_x = 0, *buf_x_orig = 0;
	int *ind_y[4], *ind_x[4];

	float *ref_scale;
	float *dis_scale;

	adm_dwt_band_t ref_dwt2;
	adm_dwt_band_t dis_dwt2;

	adm_dwt_band_t decouple_r;
	adm_dwt_band_t decouple_a;

	adm_dwt_band_t csf_a;
	adm_dwt_band_t csf_f; //Store filtered coeffs

	integer_adm_dwt_band_t_s integer_ref_dwt2;
	integer_adm_dwt_band_t_s integer_dis_dwt2;

	integer_adm_dwt_band_t_s integer_decouple_r;
	integer_adm_dwt_band_t_s integer_decouple_a;

	integer_adm_dwt_band_t_s integer_csf_a;
	integer_adm_dwt_band_t_s integer_csf_f;

	integer_i4_adm_dwt_band_t_s integer_i4_ref_dwt2;
	integer_i4_adm_dwt_band_t_s integer_i4_dis_dwt2;

	integer_i4_adm_dwt_band_t_s integer_i4_decouple_r;
	integer_i4_adm_dwt_band_t_s integer_i4_decouple_a;

	integer_i4_adm_dwt_band_t_s integer_i4_csf_a;
	integer_i4_adm_dwt_band_t_s integer_i4_csf_f;

	const float *curr_ref_scale;
	const float *curr_dis_scale;

	int16_t *integer_curr_ref_scale = ref;
	int16_t *integer_curr_dis_scale = dis;

	int32_t *integer_i4_curr_ref_scale;
	int32_t *integer_i4_curr_dis_scale;

    int32_t *tmp_ref;

	int curr_ref_stride = ref_stride;
	int curr_dis_stride = dis_stride;

	int orig_h = h;

	int buf_stride = ALIGN_CEIL(((w + 1) / 2) * sizeof(float));
	size_t buf_sz_one = (size_t)buf_stride * ((h + 1) / 2);

	int ind_size_y = ALIGN_CEIL(((h + 1) / 2) * sizeof(int));
	int ind_size_x = ALIGN_CEIL(((w + 1) / 2) * sizeof(int));

	double num = 0;
	double den = 0;

	int scale;
	int ret = 1;

	// Code optimized to save on multiple buffer copies
	// hence the reduction in the number of buffers required from 35 to 17
#define NUM_BUFS_ADM 30
	if (SIZE_MAX / buf_sz_one < NUM_BUFS_ADM)
	{
		printf("error: SIZE_MAX / buf_sz_one < NUM_BUFS_ADM, buf_sz_one = %zu.\n", buf_sz_one);
		fflush(stdout);
		goto fail;
	}

	if (!(data_buf = aligned_malloc(buf_sz_one * (NUM_BUFS_ADM), MAX_ALIGN)))
	{
		printf("error: aligned_malloc failed for data_buf.\n");
		fflush(stdout);
		goto fail;
	}

    if (!(tmp_ref = aligned_malloc(w * (sizeof(int32_t) * 4), MAX_ALIGN)))
    {
        printf("error: aligned_malloc failed for tmp_ref.\n");
        fflush(stdout);
        goto fail;
    }

	data_top = (char *)data_buf;
	
	data_top = integer_init_dwt_band(&integer_ref_dwt2, data_top, buf_sz_one/2);
	data_top = integer_init_dwt_band(&integer_dis_dwt2, data_top, buf_sz_one/2);
	data_top = integer_init_dwt_band_hvd(&integer_decouple_r, data_top, buf_sz_one/2);
	data_top = integer_init_dwt_band_hvd(&integer_decouple_a, data_top, buf_sz_one/2);
	data_top = integer_init_dwt_band_hvd(&integer_csf_a, data_top, buf_sz_one/2);
	data_top = integer_init_dwt_band_hvd(&integer_csf_f, data_top, buf_sz_one/2);
	
	data_top = integer_i4_init_dwt_band(&integer_i4_ref_dwt2, data_top, buf_sz_one);
	data_top = integer_i4_init_dwt_band(&integer_i4_dis_dwt2, data_top, buf_sz_one);
	data_top = integer_i4_init_dwt_band_hvd(&integer_i4_decouple_r, data_top, buf_sz_one);
	data_top = integer_i4_init_dwt_band_hvd(&integer_i4_decouple_a, data_top, buf_sz_one);
	data_top = integer_i4_init_dwt_band_hvd(&integer_i4_csf_a, data_top, buf_sz_one);
	data_top = integer_i4_init_dwt_band_hvd(&integer_i4_csf_f, data_top, buf_sz_one);

	if (!(buf_y_orig = aligned_malloc(ind_size_y * 4, MAX_ALIGN)))
	{
		printf("error: aligned_malloc failed for ind_buf_y.\n");
		fflush(stdout);
		goto fail;
	}
	ind_buf_y = buf_y_orig;
	ind_y[0] = (int*)ind_buf_y; ind_buf_y += ind_size_y;
	ind_y[1] = (int*)ind_buf_y; ind_buf_y += ind_size_y;
	ind_y[2] = (int*)ind_buf_y; ind_buf_y += ind_size_y;
	ind_y[3] = (int*)ind_buf_y; ind_buf_y += ind_size_y;

	if (!(buf_x_orig = aligned_malloc(ind_size_x * 4, MAX_ALIGN)))
	{
		printf("error: aligned_malloc failed for ind_buf_x.\n");
		fflush(stdout);
		goto fail;
	}
	ind_buf_x = buf_x_orig;
	ind_x[0] = (int*)ind_buf_x; ind_buf_x += ind_size_x;
	ind_x[1] = (int*)ind_buf_x; ind_buf_x += ind_size_x;
	ind_x[2] = (int*)ind_buf_x; ind_buf_x += ind_size_x;
	ind_x[3] = (int*)ind_buf_x; ind_buf_x += ind_size_x;

	for (scale = 0; scale < 4; ++scale) {
#ifdef ADM_OPT_DEBUG_DUMP
		char pathbuf[256];
#endif
		float num_scale = 0.0;
		float den_scale = 0.0;

        dwt2_src_indices_filt_s(ind_y, ind_x, w, h);
		if(scale==0)
		{
			integer_adm_dwt2_s(integer_curr_ref_scale, &integer_ref_dwt2, ind_y, ind_x, w, h, curr_ref_stride, buf_stride, tmp_ref, inp_size_bits);
			integer_adm_dwt2_s(integer_curr_dis_scale, &integer_dis_dwt2, ind_y, ind_x, w, h, curr_dis_stride, buf_stride, tmp_ref, inp_size_bits);

			i16_to_i32(&integer_ref_dwt2, &integer_i4_ref_dwt2, w, h, buf_stride, buf_stride);
			i16_to_i32(&integer_dis_dwt2, &integer_i4_dis_dwt2, w, h, buf_stride, buf_stride);

			w = (w + 1) / 2;
			h = (h + 1) / 2;

			integer_adm_decouple_s(&integer_ref_dwt2, &integer_dis_dwt2, &integer_decouple_r, &integer_decouple_a, w, h, buf_stride, buf_stride, buf_stride, buf_stride, border_factor);

			den_scale = integer_adm_csf_den_scale_s(&integer_ref_dwt2, orig_h, scale, w, h, buf_stride, border_factor);

			integer_adm_csf_s(&integer_decouple_a, &integer_csf_a, &integer_csf_f, orig_h, scale, w, h, buf_stride, buf_stride, border_factor);

			num_scale = integer_adm_cm_s(&integer_decouple_r, &integer_csf_f, &integer_csf_a, w, h, buf_stride, buf_stride, buf_stride, border_factor, scale);

		}
		else
		{
            integer_adm_dwt2_scale123_combined_s(integer_i4_curr_ref_scale, &integer_i4_ref_dwt2, integer_i4_curr_dis_scale, &integer_i4_dis_dwt2, ind_y, ind_x, w, h, curr_ref_stride, curr_dis_stride, buf_stride, scale, tmp_ref);

			w = (w + 1) / 2;
			h = (h + 1) / 2;

			integer_adm_decouple_scale123_s(&integer_i4_ref_dwt2, &integer_i4_dis_dwt2, &integer_i4_decouple_r, &integer_i4_decouple_a, w, h, buf_stride, buf_stride, buf_stride, buf_stride, border_factor, scale);

			den_scale = integer_adm_csf_den_scale123_s(&integer_i4_ref_dwt2, orig_h, scale, w, h, buf_stride, border_factor);

			integer_i4_adm_csf_s(&integer_i4_decouple_a, &integer_i4_csf_a, &integer_i4_csf_f, orig_h, scale, w, h, buf_stride, buf_stride, border_factor);

			num_scale = integer_i4_adm_cm_s(&integer_i4_decouple_r, &integer_i4_csf_f, &integer_i4_csf_a, w, h, buf_stride, buf_stride, buf_stride, border_factor, scale);

		}
#ifdef ADM_OPT_DEBUG_DUMP
		sprintf(pathbuf, "stage/ref[%d]_a.yuv", scale);
		write_image(pathbuf, ref_dwt2.band_a, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/ref[%d]_h.yuv", scale);
		write_image(pathbuf, ref_dwt2.band_h, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/ref[%d]_v.yuv", scale);
		write_image(pathbuf, ref_dwt2.band_v, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/ref[%d]_d.yuv", scale);
		write_image(pathbuf, ref_dwt2.band_d, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/dis[%d]_a.yuv", scale);
		write_image(pathbuf, dis_dwt2.band_a, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/dis[%d]_h.yuv", scale);
		write_image(pathbuf, dis_dwt2.band_h, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/dis[%d]_v.yuv", scale);
		write_image(pathbuf, dis_dwt2.band_v, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/dis[%d]_d.yuv", scale);
		write_image(pathbuf, dis_dwt2.band_d, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/r[%d]_h.yuv", scale);
		write_image(pathbuf, decouple_r.band_h, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/r[%d]_v.yuv", scale);
		write_image(pathbuf, decouple_r.band_v, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/r[%d]_d.yuv", scale);
		write_image(pathbuf, decouple_r.band_d, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/a[%d]_h.yuv", scale);
		write_image(pathbuf, decouple_a.band_h, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/a[%d]_v.yuv", scale);
		write_image(pathbuf, decouple_a.band_v, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/a[%d]_d.yuv", scale);
		write_image(pathbuf, decouple_a.band_d, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/csf_a[%d]_h.yuv", scale);
		write_image(pathbuf, csf_a.band_h, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/csf_a[%d]_v.yuv", scale);
		write_image(pathbuf, csf_a.band_v, w, h, buf_stride, sizeof(float));

		sprintf(pathbuf, "stage/csf_a[%d]_d.yuv", scale);
		write_image(pathbuf, csf_a.band_d, w, h, buf_stride, sizeof(float));

#endif

		num += num_scale;
		den += den_scale;

		integer_i4_curr_ref_scale = integer_i4_ref_dwt2.band_a;
		integer_i4_curr_dis_scale = integer_i4_dis_dwt2.band_a;

		curr_ref_stride = buf_stride;
		curr_dis_stride = buf_stride;

#ifdef ADM_OPT_DEBUG_DUMP
		PRINTF("num: %f\n", num);
		PRINTF("den: %f\n", den);
#endif
		scores[2 * scale + 0] = num_scale;
		scores[2 * scale + 1] = den_scale;
	}

	num = num < numden_limit ? 0 : num;
	den = den < numden_limit ? 0 : den;

	if (den == 0.0)
	{
		*score = 1.0f;
	}
	else
	{
		*score = num / den;
	}
	*score_num = num;
	*score_den = den;

	ret = 0;

fail:
	aligned_free(data_buf);
	aligned_free(buf_y_orig);
	aligned_free(buf_x_orig);
	aligned_free(tmp_ref);
	return ret;
}
