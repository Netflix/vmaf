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
 */

#ifndef _IQA_H_
#define _IQA_H_

#include "iqa_os.h"

/**
 * Allows fine-grain control of the SSIM algorithm.
 */
struct iqa_ssim_args {
    float alpha;    /**< luminance exponent */
    float beta;     /**< contrast exponent */
    float gamma;    /**< structure exponent */
    int L;          /**< dynamic range (2^8 - 1)*/
    float K1;       /**< stabilization constant 1 */
    float K2;       /**< stabilization constant 2 */
    int f;          /**< scale factor. 0=default scaling, 1=no scaling */
};

/**
 * Allows fine-grain control of the MS-SSIM algorithm.
 */
struct iqa_ms_ssim_args {
    int wang;             /**< 1=original algorithm by Wang, et al. 0=MS-SSIM* by Rouse/Hemami (default). */
    int gaussian;         /**< 1=11x11 Gaussian window (default). 0=8x8 linear window. */
    int scales;           /**< Number of scaled images to use. Default is 5. */
    const float *alphas;  /**< Pointer to array of alpha values for each scale. Required if 'scales' isn't 5. */
    const float *betas;   /**< Pointer to array of beta values for each scale. Required if 'scales' isn't 5. */
    const float *gammas;  /**< Pointer to array of gamma values for each scale. Required if 'scales' isn't 5. */
};

#endif /*_IQA_H_*/
