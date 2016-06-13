/*
 * Copyright (c) 2011, Tom Distler (http://tdistler.com)
 * All rights reserved.
 *
 * The BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * - Neither the name of the tdistler.com nor the names of its contributors may
 *   be used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * (06/10/2016) Updated by zli-nflx (zli@netflix.com) to output mean luminence,
 * contrast and structure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h> /* zli-nflx */

#include "iqa.h"
#include "convolve.h"
#include "decimate.h"
#include "math_utils.h"
#include "ssim_tools.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

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

/* _calc_luminance */
IQA_INLINE static double _calc_luminance(float mu1, float mu2, float C1, float alpha)
{
    double result;
    float sign;
    /* For MS-SSIM* */
    if (C1 == 0 && mu1*mu1 == 0 && mu2*mu2 == 0)
        return 1.0;
    result = (2.0 * mu1 * mu2 + C1) / (mu1*mu1 + mu2*mu2 + C1);
    if (alpha == 1.0f)
        return result;
    sign = result < 0.0 ? -1.0f : 1.0f;
    return sign * pow(fabs(result),(double)alpha);
}

/* _calc_contrast */
IQA_INLINE static double _calc_contrast(double sigma_comb_12, float sigma1_sqd, float sigma2_sqd, float C2, float beta)
{
    double result;
    float sign;
    /* For MS-SSIM* */
    if (C2 == 0 && sigma1_sqd + sigma2_sqd == 0)
        return 1.0;
    result = (2.0 * sigma_comb_12 + C2) / (sigma1_sqd + sigma2_sqd + C2);
    if (beta == 1.0f)
        return result;
    sign = result < 0.0 ? -1.0f : 1.0f;
    return sign * pow(fabs(result),(double)beta);
}

/* _calc_structure */
IQA_INLINE static double _calc_structure(float sigma_12, double sigma_comb_12, float sigma1, float sigma2, float C3, float gamma)
{
    double result;
    float sign;
    /* For MS-SSIM* */
    if (C3 == 0 && sigma_comb_12 == 0) {
        if (sigma1 == 0 && sigma2 == 0)
            return 1.0;
        else if (sigma1 == 0 || sigma2 == 0)
            return 0.0;
    }
    result = (sigma_12 + C3) / (sigma_comb_12 + C3);
    if (gamma == 1.0f)
        return result;
    sign = result < 0.0 ? -1.0f : 1.0f;
    return sign * pow(fabs(result),(double)gamma);
}

/* _iqa_ssim */
float _iqa_ssim(float *ref, float *cmp, int w, int h, const struct _kernel *k,
		const struct _map_reduce *mr, const struct iqa_ssim_args *args
		, float *l_mean, float *c_mean, float *s_mean /* zli-nflx */
		)
{
    float alpha=1.0f, beta=1.0f, gamma=1.0f;
    int L=255;
    float K1=0.01f, K2=0.03f;
    float C1,C2,C3;
    int x,y,offset;
    float *ref_mu,*cmp_mu,*ref_sigma_sqd,*cmp_sigma_sqd,*sigma_both;
    double ssim_sum;
    // double numerator, denominator; /* zli-nflx */
    double luminance_comp, contrast_comp, structure_comp, sigma_root;
    struct _ssim_int sint;
    double l_sum, c_sum, s_sum, l, c, s, sigma_ref_sigma_cmp; /* zli-nflx */

    assert(!args); /* zli-nflx: for now only works for default case */

    /* Initialize algorithm parameters */
    if (args) {
        if (!mr)
            return INFINITY;
        alpha = args->alpha;
        beta  = args->beta;
        gamma = args->gamma;
        L     = args->L;
        K1    = args->K1;
        K2    = args->K2;
    }
    C1 = (K1*L)*(K1*L);
    C2 = (K2*L)*(K2*L);
    C3 = C2 / 2.0f;

    ref_mu = (float*)malloc(w*h*sizeof(float));
    cmp_mu = (float*)malloc(w*h*sizeof(float));
    ref_sigma_sqd = (float*)malloc(w*h*sizeof(float));
    cmp_sigma_sqd = (float*)malloc(w*h*sizeof(float));
    sigma_both = (float*)malloc(w*h*sizeof(float));
    if (!ref_mu || !cmp_mu || !ref_sigma_sqd || !cmp_sigma_sqd || !sigma_both) {
        if (ref_mu) free(ref_mu);
        if (cmp_mu) free(cmp_mu);
        if (ref_sigma_sqd) free(ref_sigma_sqd);
        if (cmp_sigma_sqd) free(cmp_sigma_sqd);
        if (sigma_both) free(sigma_both);
        return INFINITY;
    }

    /* Calculate mean */
    _iqa_convolve(ref, w, h, k, ref_mu, 0, 0);
    _iqa_convolve(cmp, w, h, k, cmp_mu, 0, 0);

    for (y=0; y<h; ++y) {
        offset = y*w;
        for (x=0; x<w; ++x, ++offset) {
            ref_sigma_sqd[offset] = ref[offset] * ref[offset];
            cmp_sigma_sqd[offset] = cmp[offset] * cmp[offset];
            sigma_both[offset] = ref[offset] * cmp[offset];
        }
    }

    /* Calculate sigma */
    _iqa_convolve(ref_sigma_sqd, w, h, k, 0, 0, 0);
    _iqa_convolve(cmp_sigma_sqd, w, h, k, 0, 0, 0);
    _iqa_convolve(sigma_both, w, h, k, 0, &w, &h); /* Update the width and height */

    /* The convolution results are smaller by the kernel width and height */
    for (y=0; y<h; ++y) {
        offset = y*w;
        for (x=0; x<w; ++x, ++offset) {
            ref_sigma_sqd[offset] -= ref_mu[offset] * ref_mu[offset];
            cmp_sigma_sqd[offset] -= cmp_mu[offset] * cmp_mu[offset];
            sigma_both[offset] -= ref_mu[offset] * cmp_mu[offset];
        }
    }

    ssim_sum = 0.0;
    l_sum = 0.0; /* zli-nflx */
    c_sum = 0.0; /* zli-nflx */
    s_sum = 0.0; /* zli-nflx */
    for (y=0; y<h; ++y) {
        offset = y*w;
        for (x=0; x<w; ++x, ++offset) {

            if (!args) {

            	/* The default case */
                // numerator   = (2.0 * ref_mu[offset] * cmp_mu[offset] + C1) * (2.0 * sigma_both[offset] + C2);
                // denominator = (ref_mu[offset]*ref_mu[offset] + cmp_mu[offset]*cmp_mu[offset] + C1) *
                //     (ref_sigma_sqd[offset] + cmp_sigma_sqd[offset] + C2);
                // ssim_sum += numerator / denominator;

                /* zli-nflx: */
                sigma_ref_sigma_cmp = sqrt(ref_sigma_sqd[offset] * cmp_sigma_sqd[offset]);
                l = (2.0 * ref_mu[offset] * cmp_mu[offset] + C1) / (ref_mu[offset]*ref_mu[offset] + cmp_mu[offset]*cmp_mu[offset] + C1);
                c = (2.0 * sigma_ref_sigma_cmp + C2) /  (ref_sigma_sqd[offset] + cmp_sigma_sqd[offset] + C2);
                s = (sigma_both[offset] + C2 / 2.0) / (sigma_ref_sigma_cmp + C2 / 2.0);
                ssim_sum += l * c * s;
                l_sum += l;
                c_sum += c;
                s_sum += s;
            }
            else {
                /* User tweaked alpha, beta, or gamma */

                /* passing a negative number to sqrt() cause a domain error */
                if (ref_sigma_sqd[offset] < 0.0f)
                    ref_sigma_sqd[offset] = 0.0f;
                if (cmp_sigma_sqd[offset] < 0.0f)
                    cmp_sigma_sqd[offset] = 0.0f;
                sigma_root = sqrt(ref_sigma_sqd[offset] * cmp_sigma_sqd[offset]);

                luminance_comp = _calc_luminance(ref_mu[offset], cmp_mu[offset], C1, alpha);
                contrast_comp  = _calc_contrast(sigma_root, ref_sigma_sqd[offset], cmp_sigma_sqd[offset], C2, beta);
                structure_comp = _calc_structure(sigma_both[offset], sigma_root, ref_sigma_sqd[offset], cmp_sigma_sqd[offset], C3, gamma);

                sint.l = luminance_comp;
                sint.c = contrast_comp;
                sint.s = structure_comp;

                if (mr->map(&sint, mr->context))
                    return INFINITY;
            }
        }
    }

    free(ref_mu);
    free(cmp_mu);
    free(ref_sigma_sqd);
    free(cmp_sigma_sqd);
    free(sigma_both);

    if (!args) {
    	*l_mean = (float)(l_sum / (double)(w*h)); /* zli-nflx */
    	*c_mean = (float)(c_sum / (double)(w*h)); /* zli-nflx */
    	*s_mean = (float)(s_sum / (double)(w*h)); /* zli-nflx */
        return (float)(ssim_sum / (double)(w*h));
    }
    return mr->reduce(w, h, mr->context);
}

int compute_ssim(const number_t *ref, const number_t *cmp, int w, int h,
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
    window.w = window.h = SQUARE_LEN;
    window.normalized = 1;
    window.bnd_opt = KBND_SYMMETRIC;
    if (gaussian) {
        window.kernel = (float*)g_gaussian_window;
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
        if (!low_pass.kernel) {
            free(ref_f);
            free(cmp_f);
            printf("error: unable to malloc low-pass filter kernel.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        low_pass.w = low_pass.h = scale;
        low_pass.normalized = 0;
        low_pass.bnd_opt = KBND_SYMMETRIC;
        for (offset=0; offset<scale*scale; ++offset)
            low_pass.kernel[offset] = 1.0f/(scale*scale);

        /* resample */
        if (_iqa_decimate(ref_f, w, h, scale, &low_pass, 0, 0, 0) ||
            _iqa_decimate(cmp_f, w, h, scale, &low_pass, 0, &w, &h)) { /* update w/h */
            free(ref_f);
            free(cmp_f);
            free(low_pass.kernel);
            printf("error: decimation fails on ref_f or cmp_f.\n");
            fflush(stdout);
            goto fail_or_end;
        }
        free(low_pass.kernel);
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

/* Low-pass filter for down-sampling (9/7 biorthogonal wavelet filter) */
#define LPF_LEN 9
static const float g_lpf[LPF_LEN][LPF_LEN] = {
   { 0.000714f,-0.000450f,-0.002090f, 0.007132f, 0.016114f, 0.007132f,-0.002090f,-0.000450f, 0.000714f},
   {-0.000450f, 0.000283f, 0.001316f,-0.004490f,-0.010146f,-0.004490f, 0.001316f, 0.000283f,-0.000450f},
   {-0.002090f, 0.001316f, 0.006115f,-0.020867f,-0.047149f,-0.020867f, 0.006115f, 0.001316f,-0.002090f},
   { 0.007132f,-0.004490f,-0.020867f, 0.071207f, 0.160885f, 0.071207f,-0.020867f,-0.004490f, 0.007132f},
   { 0.016114f,-0.010146f,-0.047149f, 0.160885f, 0.363505f, 0.160885f,-0.047149f,-0.010146f, 0.016114f},
   { 0.007132f,-0.004490f,-0.020867f, 0.071207f, 0.160885f, 0.071207f,-0.020867f,-0.004490f, 0.007132f},
   {-0.002090f, 0.001316f, 0.006115f,-0.020867f,-0.047149f,-0.020867f, 0.006115f, 0.001316f,-0.002090f},
   {-0.000450f, 0.000283f, 0.001316f,-0.004490f,-0.010146f,-0.004490f, 0.001316f, 0.000283f,-0.000450f},
   { 0.000714f,-0.000450f,-0.002090f, 0.007132f, 0.016114f, 0.007132f,-0.002090f,-0.000450f, 0.000714f},
};

/* Alpha, beta, and gamma values for each scale */
static float g_alphas[] = { 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f };
static float g_betas[]  = { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f };
static float g_gammas[] = { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f };

struct _context {
    double l;  /* Luminance */
    double c;  /* Contrast */
    double s;  /* Structure */
    float alpha;
    float beta;
    float gamma;
};

/* Called for each pixel */
int _ms_ssim_map(const struct _ssim_int *si, void *ctx)
{
    struct _context *ms_ctx = (struct _context*)ctx;
    ms_ctx->l += si->l;
    ms_ctx->c += si->c;
    ms_ctx->s += si->s;
    return 0;
}

/* Called to calculate the final result */
float _ms_ssim_reduce(int w, int h, void *ctx)
{
    double size = (double)(w*h);
    struct _context *ms_ctx = (struct _context*)ctx;
    ms_ctx->l = pow(ms_ctx->l / size, (double)ms_ctx->alpha);
    ms_ctx->c = pow(ms_ctx->c / size, (double)ms_ctx->beta);
    ms_ctx->s = pow(fabs(ms_ctx->s / size), (double)ms_ctx->gamma);
    return (float)(ms_ctx->l * ms_ctx->c * ms_ctx->s);
}

/* Releases the scaled buffers */
void _free_buffers(float **buf, int scales)
{
    int idx;
    for (idx=0; idx<scales; ++idx)
        free(buf[idx]);
}

/* Allocates the scaled buffers. If error, all buffers are free'd */
int _alloc_buffers(float **buf, int w, int h, int scales)
{
    int idx;
    int cur_w = w;
    int cur_h = h;
    for (idx=0; idx<scales; ++idx) {
        buf[idx] = (float*)malloc(cur_w*cur_h*sizeof(float));
        if (!buf[idx]) {
            _free_buffers(buf, idx);
            return 1;
        }
        cur_w = cur_w/2 + (cur_w&1);
        cur_h = cur_h/2 + (cur_h&1);
    }
    return 0;
}

int compute_ms_ssim(const number_t *ref, const number_t *cmp, int w, int h,
		int ref_stride, int cmp_stride, double *score,
		double* l_scores, double* c_scores, double* s_scores)
{

	int ret = 1;

    int wang=1; /* set default to wang's ms_ssim */
    int scales=SCALES;
    int gauss=1;
    const float *alphas=g_alphas, *betas=g_betas, *gammas=g_gammas;
    int idx,x,y,cur_w,cur_h;
    int offset,src_offset;
    float **ref_imgs, **cmp_imgs; /* Array of pointers to scaled images */
    double msssim;
    float l, c, s;
    struct _kernel lpf, window;
    struct iqa_ssim_args s_args;
    struct _map_reduce mr;
    struct _context ms_ctx;

    /* check stride */
	int stride = ref_stride; /* stride in bytes */
	if (stride != cmp_stride)
	{
		printf("error: for ms_ssim, ref_stride (%d) != dis_stride (%d) bytes.\n", ref_stride, cmp_stride);
		fflush(stdout);
		goto fail_or_end;
	}
	stride /= sizeof(float); /* stride_ in pixels */

	/* specify some default parameters */
	const struct iqa_ms_ssim_args *args = 0; /* 0 for default */

    /* initialize algorithm parameters */
    if (args) {
        wang   = args->wang;
        gauss  = args->gaussian;
        scales = args->scales;
        if (args->alphas)
            alphas = args->alphas;
        if (args->betas)
            betas  = args->betas;
        if (args->gammas)
            gammas = args->gammas;
    }

    /* make sure we won't scale below 1x1 */
    cur_w = w;
    cur_h = h;
    for (idx=0; idx<scales; ++idx) {
        if ( gauss ? cur_w<GAUSSIAN_LEN || cur_h<GAUSSIAN_LEN : cur_w<LPF_LEN || cur_h<LPF_LEN )
        {
    		printf("error: scale below 1x1!\n");
            goto fail_or_end;
        }
        cur_w /= 2;
        cur_h /= 2;
    }

    window.kernel = (float*)g_square_window;
    window.w = window.h = SQUARE_LEN;
    window.normalized = 1;
    window.bnd_opt = KBND_SYMMETRIC;
    if (gauss) {
        window.kernel = (float*)g_gaussian_window;
        window.w = window.h = GAUSSIAN_LEN;
    }

    mr.map     = _ms_ssim_map;
    mr.reduce  = _ms_ssim_reduce;

    /* allocate the scaled image buffers */
    ref_imgs = (float**)malloc(scales*sizeof(float*));
    cmp_imgs = (float**)malloc(scales*sizeof(float*));
    if (!ref_imgs || !cmp_imgs) {
        if (ref_imgs) free(ref_imgs);
        if (cmp_imgs) free(cmp_imgs);
		printf("error: unable to malloc ref_imgs or cmp_imgs.\n");
		fflush(stdout);
		goto fail_or_end;
    }
    if (_alloc_buffers(ref_imgs, w, h, scales)) {
        free(ref_imgs);
        free(cmp_imgs);
		printf("error: unable to _alloc_buffers on ref_imgs.\n");
		fflush(stdout);
		goto fail_or_end;
    }
    if (_alloc_buffers(cmp_imgs, w, h, scales)) {
        _free_buffers(ref_imgs, scales);
        free(ref_imgs);
        free(cmp_imgs);
		printf("error: unable to _alloc_buffers on cmp_imgs.\n");
		fflush(stdout);
		goto fail_or_end;
    }

    /* copy original images into first scale buffer, forcing stride = width. */
    for (y=0; y<h; ++y) {
        src_offset = y * stride;
        offset = y * w;
        for (x=0; x<w; ++x, ++offset, ++src_offset) {
            ref_imgs[0][offset] = (float)ref[src_offset];
            cmp_imgs[0][offset] = (float)cmp[src_offset];
        }
    }

    /* create scaled versions of the images */
    cur_w=w;
    cur_h=h;
    lpf.kernel = (float*)g_lpf;
    lpf.w = lpf.h = LPF_LEN;
    lpf.normalized = 1;
    lpf.bnd_opt = KBND_SYMMETRIC;
    for (idx=1; idx<scales; ++idx) {
        if (_iqa_decimate(ref_imgs[idx-1], cur_w, cur_h, 2, &lpf, ref_imgs[idx], 0, 0) ||
            _iqa_decimate(cmp_imgs[idx-1], cur_w, cur_h, 2, &lpf, cmp_imgs[idx], &cur_w, &cur_h))
        {
            _free_buffers(ref_imgs, scales);
            _free_buffers(cmp_imgs, scales);
            free(ref_imgs);
            free(cmp_imgs);
    		printf("error: decimation fails on ref_imgs or cmp_imgs.\n");
    		fflush(stdout);
    		goto fail_or_end;
        }
    }

    cur_w=w;
    cur_h=h;
    msssim = 1.0;
    for (idx=0; idx<scales; ++idx) {

        ms_ctx.l = 0;
        ms_ctx.c = 0;
        ms_ctx.s = 0;
        ms_ctx.alpha = alphas[idx];
        ms_ctx.beta  = betas[idx];
        ms_ctx.gamma = gammas[idx];

        if (!wang) {
            /* MS-SSIM* (Rouse/Hemami) */
            s_args.alpha = 1.0f;
            s_args.beta  = 1.0f;
            s_args.gamma = 1.0f;
            s_args.K1 = 0.0f; /* Force stabilization constants to 0 */
            s_args.K2 = 0.0f;
            s_args.L  = 255;
            s_args.f  = 1; /* Don't resize */
            mr.context = &ms_ctx;
            _iqa_ssim(ref_imgs[idx], cmp_imgs[idx], cur_w, cur_h, &window, &mr, &s_args, &l, &c, &s);
        }
        else {
            /* MS-SSIM (Wang) */
        	/*
			s_args.alpha = 1.0f;
			s_args.beta  = 1.0f;
			s_args.gamma = 1.0f;
			s_args.K1 = 0.01f;
			s_args.K2 = 0.03f;
			s_args.L  = 255;
			s_args.f  = 1; // Don't resize
			mr.context = &ms_ctx;
			msssim *= _iqa_ssim(ref_imgs[idx], cmp_imgs[idx], cur_w, cur_h, &window, &mr, &s_args, &l, &c, &s);
			*/

        	/* above is equivalent to passing default parameter: */
        	_iqa_ssim(ref_imgs[idx], cmp_imgs[idx], cur_w, cur_h, &window, NULL, NULL, &l, &c, &s);

        }

        msssim *= pow(l, alphas[idx]) * pow(c, betas[idx]) * pow(s, gammas[idx]);
        l_scores[idx] = l;
        c_scores[idx] = c;
        s_scores[idx] = s;

        if (msssim == INFINITY) {
            _free_buffers(ref_imgs, scales);
            _free_buffers(cmp_imgs, scales);
            free(ref_imgs);
            free(cmp_imgs);
    		printf("error: ms_ssim is INFINITY.\n");
    		fflush(stdout);
            goto fail_or_end;
        }
        cur_w = cur_w/2 + (cur_w&1);
        cur_h = cur_h/2 + (cur_h&1);
    }

    _free_buffers(ref_imgs, scales);
    _free_buffers(cmp_imgs, scales);
    free(ref_imgs);
    free(cmp_imgs);

	*score = msssim;

	ret = 0;
fail_or_end:
	return ret;

}
