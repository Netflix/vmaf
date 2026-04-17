/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
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

#include <math.h>
#include "common/macros.h"
#include <errno.h>

#pragma once

#ifndef BARTEN_CSF_TOOLS_H_
#define BARTEN_CSF_TOOLS_H_

static float linear_interpolate(float left_position, float left_value,
                                float right_position, float right_value,
                                float sample_position) {
    return left_value + ((right_value - left_value) / (right_position - left_position)) * (sample_position - left_position);
}

/*
 * There were 6 luminance levels (in cd/m2) that were measured in the CSF experiments of HDR-VDP2
 * Note: HDR-VDP2 paper has 5 levels, but version 2.2.1 was updated with data for 6 levels:
 * 0.002, 0.02, 0.2, 2, 20, 150
 */
static const float barten_csf_param_anchors[6] = { 0.002, 0.02, 0.2, 2, 20, 150 };

static const float barten_csf_params[6][5] = {
    { 0.0160737, 0.991265, 3.74038, 0.50722, 4.46044  },
    { 0.383873, 0.800889, 3.54104, 0.682505, 4.94958  },
    { 0.929301, 0.476505, 4.37453, 0.750315, 5.28678  },
    { 1.29776, 0.405782, 4.40602, 0.935314, 5.61425  },
    { 1.49222, 0.334278, 3.79542, 1.07327, 6.4635 },
    { 1.46213, 0.394533, 2.7755, 1.16577, 7.45665 }
};

static const float barten_csf_sa[4] = { 30.162, 4.0627, 1.6596, 0.2712 };
static const float barten_mtf_params_a[4] = { 0.424838596301290, 0.572435103936480, 0.000167576239164937, 0.00255872352306433 };
static const float barten_mtf_params_b[4] = { 0.028, 0.37, 37, 360 };

/* Luminance-dependent component of Barten's CSF */
static float barten_rod_cone_sens(float luminance_level)
{
    float cvi_sens_drop = barten_csf_sa[1];  // p6 in the paper
    float cvi_trans_slope = barten_csf_sa[2];  // p7 in the paper
    float cvi_low_slope = barten_csf_sa[3];  // p8 in the paper
    return barten_csf_sa[0] * pow(pow(cvi_sens_drop / luminance_level, cvi_trans_slope) + 1, - cvi_low_slope);
}

/*  MTF portion of Barten's CSF */
static float barten_mtf(float spatial_frequency)
{
    float mtf = 0.0;
    for (int i = 0; i <= 3; i++)
        mtf = mtf + barten_mtf_params_a[i] * exp(- barten_mtf_params_b[i] * spatial_frequency);
    return mtf;
}

#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

/*
 * Simplified version of Barten's CSF used in HDR-VDP2 (ported from Matlab version HDR-VDP 2.2.1)
 * HDR-VDP-2: A calibrated visual metric for visibility and quality predictions in all luminance conditions
 * ACM Transactions on Graphics, Volume 30, Issue 4, July 2011, Article No.: 40, pp 1–14
 * Rafal Mantiuk, Kil Joong Kim, Allan G. Rempel, Wolfgang Heidrich
 */
static FORCE_INLINE inline float barten_csf(int lambda, double adm_norm_view_dist,
                                            int adm_ref_display_height, double adm_csf_lum_level, double adm_csf_scale)
{
    /* This is the display visual resolution (DVR), in pixels/degree of visual angle. It should be ~56.55. */
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;
    /* This is the nominal spatial frequency for each DWT level; first level (level = 0) is half of the DVR. */
    float spatial_frequency = r / pow(2, lambda + 1);

    double clamped_lum = CLAMP(adm_csf_lum_level, barten_csf_param_anchors[0], barten_csf_param_anchors[5]);

    int left_lum_index = 0, right_lum_index = 0;
    int i = 0;
    while (i < 5) {
        if (clamped_lum >= barten_csf_param_anchors[i] && clamped_lum <= barten_csf_param_anchors[i + 1]) {
            left_lum_index = i;
            right_lum_index = i + 1;
            break;
        }
        i++;
    }

    float left_lum = barten_csf_param_anchors[left_lum_index];
    float right_lum = barten_csf_param_anchors[right_lum_index];

    float left_position = log10(left_lum);
    float right_position = log10(right_lum);
    float sample_position = log10(clamped_lum);

    float p_0 = linear_interpolate(left_position, barten_csf_params[left_lum_index][1], right_position,
                                   barten_csf_params[right_lum_index][1], sample_position);
    float p_1 = linear_interpolate(left_position, barten_csf_params[left_lum_index][2], right_position,
                                   barten_csf_params[right_lum_index][2], sample_position);
    float p_2 = linear_interpolate(left_position, barten_csf_params[left_lum_index][3], right_position,
                                   barten_csf_params[right_lum_index][3], sample_position);
    float p_3 = linear_interpolate(left_position, barten_csf_params[left_lum_index][4], right_position,
                                   barten_csf_params[right_lum_index][4], sample_position);

    // these values can be derived by the matlab code:
    // metric_par = hdrvdp_parse_options({})
    // hdrvdp_ncsf(rho, lum, metric_par); (any rho works)
    float a = 1.0f + pow(p_0 * spatial_frequency, p_1);
    float b = 1.0f / pow(1 - exp(- pow(spatial_frequency / 7, 2)), p_2);

    /* neural contrast sensitivity function */
    float csf = p_3 / pow(a * b, 0.5);

    /* return entire CSF */
    return csf * barten_mtf(spatial_frequency) * barten_rod_cone_sens(adm_csf_lum_level) * adm_csf_scale;
}

static const float BLENDED_CSF_1080_3H[2][4] = {
    {0.01183, 0.025026, 0.04295, 0.058621},
    {0.004302, 0.011778, 0.023918, 0.035901}
};

static const float BLENDED_CSF_1080_5H[2][4] = {
    {0.004212, 0.014809, 0.029642, 0.047464},
    {0.000984, 0.005852, 0.0146, 0.027574}
};

static const float BLENDED_CSF_2160_3H[2][4] = {
    {0.00226, 0.01183, 0.025026, 0.04295},
    {0.000479, 0.004302, 0.011778, 0.023918}
};

static const float BLENDED_CSF_2160_5H[2][4] = {
    {0.000092, 0.004212, 0.014809, 0.029642},
    {0.000050, 0.000984, 0.005852, 0.0146}
};

static const float BLENDED_CSF_720_3H[2][4] = {
    {0.018715, 0.035637, 0.052798, 0.061509},
    {0.007999, 0.018396, 0.031851, 0.037718}
};

static const float BLENDED_CSF_720_5H[2][4] = {
    {0.010144, 0.022561, 0.040309, 0.05672},
    {0.003463, 0.010282, 0.021839, 0.034641}
};

static const float BLENDED_CSF_480_3H[2][4] = {
    {0.027961, 0.045875, 0.060275, 0.056234},
    {0.013572, 0.026277, 0.036959, 0.034511}
};

static const float BLENDED_CSF_480_5H[2][4] = {
    {0.016781, 0.032822, 0.05032, 0.061594},
    {0.00691, 0.016545, 0.029917, 0.037777}
};

/* 
 * BLENDED_CSF coefficient arrays for v1.0.17+ models (with CSF bug fix)
 * These coefficients incorporate the L1 optimization scaling parameter fix
 */
static const float BLENDED_CSF_1080_3H_MAE[2][4] = {
    {0.011249, 0.022606, 0.035930, 0.045673},
    {0.004097, 0.010921, 0.021430, 0.031313}
};

static const float BLENDED_CSF_1080_5H_MAE[2][4] = {
    {0.004052, 0.013939, 0.026298, 0.038833},
    {0.000927, 0.005544, 0.013415, 0.024515}
};

static const float BLENDED_CSF_2160_3H_MAE[2][4] = {
    {0.002166, 0.011249, 0.022606, 0.035930},
    {0.000447, 0.004097, 0.010921, 0.021430}
};

static const float BLENDED_CSF_2160_5H_MAE[2][4] = {
    {0.000077, 0.004052, 0.013939, 0.026298},
    {0.000045, 0.000927, 0.005544, 0.013415}
};

static const float BLENDED_CSF_720_3H_MAE[2][4] = {
    {0.017329, 0.030870, 0.042134, 0.047410},
    {0.007509, 0.016707, 0.028072, 0.032722}
};

static const float BLENDED_CSF_720_5H_MAE[2][4] = {
    {0.009689, 0.020577, 0.034162, 0.044523},
    {0.003302, 0.009579, 0.019661, 0.030318}
};

static const float BLENDED_CSF_480_3H_MAE[2][4] = {
    {0.024969, 0.037825, 0.046676, 0.043974},
    {0.012511, 0.023424, 0.032139, 0.030166}
};

static const float BLENDED_CSF_480_5H_MAE[2][4] = {
    {0.015665, 0.028766, 0.040612, 0.047483},
    {0.006514, 0.015107, 0.026476, 0.032776}
};

/*
 * CSF function with CSF bug fix
 * Uses the corrected L1 optimization 
 */
static FORCE_INLINE inline float barten_watson_blend_csf_mae(int scale, int theta, double adm_norm_view_dist,
                                                                  int adm_ref_display_height) {
    if (adm_ref_display_height == 1080 && adm_norm_view_dist == 3.0) {
        return BLENDED_CSF_1080_3H_MAE[theta][scale];
    } else if (adm_ref_display_height == 1080 && adm_norm_view_dist == 5.0) {
        return BLENDED_CSF_1080_5H_MAE[theta][scale];
    } else if (adm_ref_display_height == 2160 && adm_norm_view_dist == 3.0) {
        return BLENDED_CSF_2160_3H_MAE[theta][scale];
    } else if (adm_ref_display_height == 2160 && adm_norm_view_dist == 5.0) {
        return BLENDED_CSF_2160_5H_MAE[theta][scale];
    } else if (adm_ref_display_height == 720 && adm_norm_view_dist == 3.0) {
        return BLENDED_CSF_720_3H_MAE[theta][scale];
    } else if (adm_ref_display_height == 720 && adm_norm_view_dist == 5.0) {
        return BLENDED_CSF_720_5H_MAE[theta][scale];
    } else if (adm_ref_display_height == 480 && adm_norm_view_dist == 3.0) {
        return BLENDED_CSF_480_3H_MAE[theta][scale];
    } else if (adm_ref_display_height == 480 && adm_norm_view_dist == 5.0) {
        return BLENDED_CSF_480_5H_MAE[theta][scale];
    } else {
        return -EINVAL;
    }
}

/*
 * Legacy Blended CSF function (backward compatibility)
 * Always uses legacy coefficients for existing models
 */
static FORCE_INLINE inline float barten_watson_blend_csf(int scale, int theta, double adm_norm_view_dist,
                                                         int adm_ref_display_height) {
    if (adm_ref_display_height == 1080 && adm_norm_view_dist == 3.0) {
        return BLENDED_CSF_1080_3H[theta][scale];
    } else if (adm_ref_display_height == 1080 && adm_norm_view_dist == 5.0) {
        return BLENDED_CSF_1080_5H[theta][scale];
    } else if (adm_ref_display_height == 2160 && adm_norm_view_dist == 3.0) {
        return BLENDED_CSF_2160_3H[theta][scale];
    } else if (adm_ref_display_height == 2160 && adm_norm_view_dist == 5.0) {
        return BLENDED_CSF_2160_5H[theta][scale];
    } else if (adm_ref_display_height == 720 && adm_norm_view_dist == 3.0) {
        return BLENDED_CSF_720_3H[theta][scale];
    } else if (adm_ref_display_height == 720 && adm_norm_view_dist == 5.0) {
        return BLENDED_CSF_720_5H[theta][scale];
    } else if (adm_ref_display_height == 480 && adm_norm_view_dist == 3.0) {
        return BLENDED_CSF_480_3H[theta][scale];
    } else if (adm_ref_display_height == 480 && adm_norm_view_dist == 5.0) {
        return BLENDED_CSF_480_5H[theta][scale];
    } else {
        return -EINVAL;
    }
}

#endif /* BARTEN_CSF_TOOLS_H_ */
