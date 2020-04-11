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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "mem.h"
#include "iqa/math_utils.h"
#include "iqa/decimate.h"
#include "iqa/ssim_tools.h"

/* _ssim_map */
int _ssim_map(const struct _ssim_int *si, void *ctx)
{
    double *ssim_sum = (double*)ctx;
    *ssim_sum += si->l * si->c * si->s;
    return 0;
}

/* _ssim_reduce */
float _ssim_reduce(int w, int h, void *ctx)
{
    double *ssim_sum = (double*)ctx;
    return (float)(*ssim_sum / (double)(w*h));
}

int compute_ssim(const float *ref, const float *cmp, int w, int h,
        int ref_stride, int cmp_stride, double *score,
        double *l_score, double *c_score, double *s_score)
{

    int ret = 1;

    int scale;
    int x,y,src_offset,offset;
    float *ref_f,*cmp_f;
    struct _kernel low_pass;
    struct _kernel window;
    float result = INFINITY;
    float l, c, s;
    double ssim_sum=0.0;
    struct _map_reduce mr;

    /* check stride */
    int stride = ref_stride; /* stride in bytes */
    if (stride != cmp_stride)
    {
        printf("error: for ssim, ref_stride (%d) != dis_stride (%d) bytes.\n", ref_stride, cmp_stride);
        fflush(stdout);
        goto fail_or_end;
    }
    stride /= sizeof(float); /* stride_ in pixels */

    /* specify some default parameters */
    const struct iqa_ssim_args *args = 0; /* 0 for default */
    int gaussian = 1; /* 0 for 8x8 square window, 1 for 11x11 circular-symmetric Gaussian window (default) */

    /* initialize algorithm parameters */
    scale = _max( 1, _round( (float)_min(w,h) / 256.0f ) );
    if (args) {
        if(args->f) {
            scale = args->f;
        }
        mr.map     = _ssim_map;
        mr.reduce  = _ssim_reduce;
        mr.context = (void*)&ssim_sum;
    }
    window.kernel = (float*)g_square_window;
    window.kernel_h = (float*)g_square_window_h;
    window.kernel_v = (float*)g_square_window_v;
    window.w = window.h = SQUARE_LEN;
    window.normalized = 1;
    window.bnd_opt = KBND_SYMMETRIC;
    if (gaussian) {
        window.kernel = (float*)g_gaussian_window;
        window.kernel_h = (float*)g_gaussian_window_h;
        window.kernel_v = (float*)g_gaussian_window_v;
        window.w = window.h = GAUSSIAN_LEN;
    }

    /* convert image values to floats, forcing stride = width. */
    ref_f = (float*)malloc(w*h*sizeof(float));
    cmp_f = (float*)malloc(w*h*sizeof(float));
    if (!ref_f || !cmp_f) {
        if (ref_f) free(ref_f);
        if (cmp_f) free(cmp_f);
        printf("error: unable to malloc ref_f or cmp_f.\n");
        fflush(stdout);
        goto fail_or_end;
    }
    for (y=0; y<h; ++y) {
        src_offset = y * stride;
        offset = y * w;
        for (x=0; x<w; ++x, ++offset, ++src_offset) {
            ref_f[offset] = (float)ref[src_offset];
            cmp_f[offset] = (float)cmp[src_offset];
        }
    }

    /* scale the images down if required */
    if (scale > 1) {
        /* generate simple low-pass filter */
        low_pass.kernel = (float*)malloc(scale*scale*sizeof(float));
        low_pass.kernel_h = (float*)malloc(scale*sizeof(float)); /* zli-nflx */
        low_pass.kernel_v = (float*)malloc(scale*sizeof(float)); /* zli-nflx */
        if (!(low_pass.kernel && low_pass.kernel_h && low_pass.kernel_v)) { /* zli-nflx */
            free(ref_f);
            free(cmp_f);
            if (low_pass.kernel) free(low_pass.kernel); /* zli-nflx */
            if (low_pass.kernel_h) free(low_pass.kernel_h); /* zli-nflx */
            if (low_pass.kernel_v) free(low_pass.kernel_v); /* zli-nflx */
            printf("error: unable to malloc low-pass filter kernel.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        low_pass.w = low_pass.h = scale;
        low_pass.normalized = 0;
        low_pass.bnd_opt = KBND_SYMMETRIC;
        for (offset=0; offset<scale*scale; ++offset)
            low_pass.kernel[offset] = 1.0f/(scale*scale);
        for (offset=0; offset<scale; ++offset)  /* zli-nflx */
            low_pass.kernel_h[offset] = 1.0f/(scale); /* zli-nflx */
        for (offset=0; offset<scale; ++offset) /* zli-nflx */
            low_pass.kernel_v[offset] = 1.0f/(scale); /* zli-nflx */

        /* resample */
        if (_iqa_decimate(ref_f, w, h, scale, &low_pass, 0, 0, 0) ||
            _iqa_decimate(cmp_f, w, h, scale, &low_pass, 0, &w, &h)) { /* update w/h */
            free(ref_f);
            free(cmp_f);
            free(low_pass.kernel);
            free(low_pass.kernel_h); /* zli-nflx */
            free(low_pass.kernel_v); /* zli-nflx */
            printf("error: decimation fails on ref_f or cmp_f.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        free(low_pass.kernel);
        free(low_pass.kernel_h); /* zli-nflx */
        free(low_pass.kernel_v); /* zli-nflx */
    }

    result = _iqa_ssim(ref_f, cmp_f, w, h, &window, &mr, args, &l, &c, &s);

    free(ref_f);
    free(cmp_f);

    *score = (double)result;
    *l_score = (double)l;
    *c_score = (double)c;
    *s_score = (double)s;

    ret = 0;
fail_or_end:
    return ret;

}

