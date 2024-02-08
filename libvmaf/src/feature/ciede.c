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

// This is in large part a port of the ciede2000 implementation from av-metrics
// (https://github.com/rust-av/av-metrics) which has the following license:

/*
The MIT License (MIT)
Copyright (c) 2019 Joshua Holmer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"
#include "mem.h"
#include "opt.h"

typedef struct CiedeState {
    VmafPicture ref;
    VmafPicture dist;
    void (*scale_chroma_planes)(VmafPicture *in, VmafPicture *out);
} CiedeState;

static void scale_chroma_planes_hbd(VmafPicture *in, VmafPicture *out)
{
    const int ss_hor = in->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = in->pix_fmt == VMAF_PIX_FMT_YUV420P;

    for (unsigned p = 0; p < 3; p++) {
        uint16_t *in_buf = in->data[p];
        uint16_t *out_buf = out->data[p];
        for (unsigned i = 0; i < out->h[p]; i++) {
            for (unsigned j = 0; j < out->w[p]; j++) {
                out_buf[j] = in_buf[(j / ((p && ss_ver) ? 2 : 1))];
            }
            in_buf += (((p && ss_hor) ? i % 2 : 1) * in->stride[p]) / 2;
            out_buf += out->stride[p] / 2;
        }
    }
}

static void scale_chroma_planes(VmafPicture *in, VmafPicture *out)
{
    const int ss_hor = in->pix_fmt != VMAF_PIX_FMT_YUV444P;
    const int ss_ver = in->pix_fmt == VMAF_PIX_FMT_YUV420P;

    for (unsigned p = 0; p < 3; p++) {
        uint8_t *in_buf = in->data[p];
        uint8_t *out_buf = out->data[p];
        for (unsigned i = 0; i < out->h[p]; i++) {
            for (unsigned j = 0; j < out->w[p]; j++) {
                out_buf[j] = in_buf[(j / ((p && ss_ver) ? 2 : 1))];
            }
            in_buf += ((p && ss_hor) ? i % 2 : 1) * in->stride[p];
            out_buf += out->stride[p];
        }
    }
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    CiedeState *s = fex->priv;
    int err = 0;

    if (pix_fmt == VMAF_PIX_FMT_YUV400P)
        return -EINVAL;

    if (pix_fmt == VMAF_PIX_FMT_YUV444P)
        return 0;

    switch (bpc) {
    case 8:
        s->scale_chroma_planes = scale_chroma_planes;
        break;
    case 10:
    case 12:
    case 16:
        s->scale_chroma_planes = scale_chroma_planes_hbd;
        break;
    default:
        return -EINVAL;
    }

    err |= vmaf_picture_alloc(&s->ref, VMAF_PIX_FMT_YUV444P, bpc, w, h);
    err |= vmaf_picture_alloc(&s->dist, VMAF_PIX_FMT_YUV444P, bpc, w, h);
    return err;
}

static float get_h_prime(const float x, const float y)
{
    if ((x == 0.0) && (y == 0.0))
        return 0.0;
    float hue_angle = atan2(x, y);
    if (hue_angle < 0.0)
        hue_angle += 2. * M_PI;
    return hue_angle;
}

static float get_delta_h_prime(const float c1, const float c2,
                               const float h_prime_1, const float h_prime_2)
{
    if ((c1 == 0.0) || (c2 == 0.0))
        return 0.0;
    if (fabsf(h_prime_1 - h_prime_2) <= M_PI)
        return h_prime_2 - h_prime_1;
    if (h_prime_2 <= h_prime_1)
        return h_prime_2 - h_prime_1 + 2. * M_PI;
    else
        return h_prime_2 - h_prime_1 - 2. * M_PI;

}

static float get_upcase_h_bar_prime(const float h_prime_1,
                                    const float h_prime_2)
{
    return fabs((h_prime_1 - h_prime_2)) > M_PI ?
        (h_prime_1 + h_prime_2 + 2.0 * M_PI) / 2.0 :
        (h_prime_1 + h_prime_2) / 2.0;
}

static float get_upcase_t(const float upcase_h_bar_prime)
{
    return 1.0 -
           0.17 * cos(upcase_h_bar_prime - M_PI / 6.0) +
           0.24 * cos(2.0 * upcase_h_bar_prime) +
           0.32 * cos(3.0 * upcase_h_bar_prime + M_PI / 30.0) -
           0.20 * cos(4.0 * upcase_h_bar_prime - 7.0 * M_PI / 20.0);
}

static float radians_to_degrees(const float radians)
{
    return radians * (180.0 / M_PI);
}

static float degrees_to_radians(const float degrees)
{
    return degrees * (M_PI / 180.0);
}

static float get_r_sub_t(const float c_bar_prime,
                         const float upcase_h_bar_prime)
{
    const float degrees =
        (radians_to_degrees(upcase_h_bar_prime) - 275.0) * (1.0 / 25.0);

    return -2.0 *
          sqrt(powf(c_bar_prime, 7) / (powf(c_bar_prime, 7) + powf(25., 7))) *
          sin(degrees_to_radians(60.0 * exp(-(powf(degrees, 2)))));
}

typedef struct LABColor {
    const float l;
    const float a;
    const float b;
} LABColor;

typedef struct KSubArgs {
    const float l;
    const float c;
    const float h;
} KSubArgs;

static float ciede2000(LABColor color_1, LABColor color_2, KSubArgs ksub)
{
    const float delta_l_prime = color_2.l - color_1.l;
    const float l_bar = (color_1.l + color_2.l) / 2;
    const float c1 = sqrt(pow(color_1.a, 2) + pow(color_1.b, 2));
    const float c2 = sqrt(pow(color_2.a, 2) + pow(color_2.b, 2));
    const float c_bar = (c1 + c2) / 2;
    const float a_prime_1 =
        color_1.a + (color_1.a / 2) *
        (1 - sqrt(pow(c_bar, 7) / (pow(c_bar, 7) + pow(25, 7))));
    const float a_prime_2 =
         color_2.a + (color_2.a / 2) *
         (1 - sqrt(pow(c_bar, 7) / (pow(c_bar, 7) + pow(25, 7))));
    const float c_prime_1 = sqrt(pow(a_prime_1, 2) + pow(color_1.b, 2));
    const float c_prime_2 = sqrt(pow(a_prime_2, 2) + pow(color_2.b, 2));
    const float c_bar_prime = (c_prime_1 + c_prime_2) / 2;
    const float delta_c_prime = c_prime_2 - c_prime_1;
    const float s_sub_l = 1. + ((0.015 * pow(l_bar - 50, 2)) /
                          sqrt(20 + pow(l_bar - 50, 2)));
    const float s_sub_c = 1. + 0.045 * c_bar_prime;
    const float h_prime_1 = get_h_prime(color_1.b, a_prime_1);
    const float h_prime_2 = get_h_prime(color_2.b, a_prime_2);
    const float delta_h_prime = get_delta_h_prime(c1, c2, h_prime_1, h_prime_2);
    const float delta_upcase_h_prime =
            2.0 * sqrt(c_prime_1 * c_prime_2) * sin(delta_h_prime / 2.0);
    const float upcase_h_bar_prime =
        get_upcase_h_bar_prime(h_prime_1, h_prime_2);
    const float upcase_t = get_upcase_t(upcase_h_bar_prime);
    const float s_sub_upcase_h = 1.0 + 0.015 * c_bar_prime * upcase_t;
    const float r_sub_t = get_r_sub_t(c_bar_prime, upcase_h_bar_prime);
    const float lightness = delta_l_prime / (ksub.l * s_sub_l);
    const float chroma  = delta_c_prime / (ksub.c * s_sub_c);
    const float hue = delta_upcase_h_prime / (ksub.h * s_sub_upcase_h);

    return sqrt(pow(lightness, 2) + pow(chroma, 2) +
                pow(hue, 2) + r_sub_t * chroma * hue);
}

static double pow_2_4(double x)
{
    return pow(x, 2.4);
}

static double rgb_to_xyz_map(double c)
{
    if (c > 10. / 255.) {
        const double A = 0.055;
        const double D = 1.0 / 1.055;
        return pow_2_4((c + A) * D);
    } else {
        const double D = 1.0 / 12.92;
        return (c * D);
    }
}

static double cbrt_approx(double c)
{
    return pow(c, 1.0 / 3.0);
}

static float xyz_to_lab_map(double c)
{
    const double KAPPA = 24389.0 / 27.0;
    const double EPSILON = 216.0 / 24389.0;

    if (c > EPSILON) {
        return cbrt_approx(c);
    } else {
        return (KAPPA * c + 16.0) * (1.0 / 116.0);
    }
}

static LABColor get_lab_color(double y, double u, double v, unsigned bpc)
{
    const double scale = 1 << (bpc - 8);

    y = (y - 16.  * scale) * (1. / (219. * scale));
    u = (u - 128. * scale) * (1. / (224. * scale));
    v = (v - 128. * scale) * (1. / (224. * scale));

    // Assumes BT.709
    double r = y + 1.28033 * v;
    double g = y - 0.21482 * u - 0.38059 * v;
    double b = y + 2.12798 * u;

    r = rgb_to_xyz_map(r);
    g = rgb_to_xyz_map(g);
    b = rgb_to_xyz_map(b);

    double x = r * 0.4124564390896921 + g * 0.357576077643909 +
              b * 0.18043748326639894;
          y = r * 0.21267285140562248 + g * 0.715152155287818 +
              b * 0.07217499330655958;
    double z = r * 0.019333895582329317 + g * 0.119192025881303 +
              b * 0.9503040785363677;

    x = xyz_to_lab_map(x * (1.0 / 0.95047));
    y = xyz_to_lab_map(y);
    z = xyz_to_lab_map(z * (1.0 / 1.08883));

    LABColor lab_color = {
        .l = (116.0 * y) - 16.0,
        .a = 500.0 * (x - y),
        .b = 200.0 * (y - z),
    };

    return lab_color;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    CiedeState *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;

    VmafPicture *ref;
    VmafPicture *dist;

    if (ref_pic->pix_fmt == VMAF_PIX_FMT_YUV444P) {
        // Reuse the provided buffers
        ref = ref_pic;
        dist = dist_pic;
    } else {
        ref = &s->ref;
        dist = &s->dist;
        s->scale_chroma_planes(ref_pic, ref);
        s->scale_chroma_planes(dist_pic, dist);
    }

    double de00_sum = 0.;
    for (unsigned i = 0; i < ref->h[0]; i++) {
        for (unsigned j = 0; j < ref->w[0]; j++) {
            float r_y, r_u, r_v, d_y, d_u, d_v;

            switch (ref->bpc) {
            case 8:
                r_y = ((uint8_t*)ref->data[0])[i * ref->stride[0] + j];
                r_u = ((uint8_t*)ref->data[1])[i * ref->stride[1] + j];
                r_v = ((uint8_t*)ref->data[2])[i * ref->stride[2] + j];
                d_y = ((uint8_t*)dist->data[0])[i * dist->stride[0] + j];
                d_u = ((uint8_t*)dist->data[1])[i * dist->stride[1] + j];
                d_v = ((uint8_t*)dist->data[2])[i * dist->stride[2] + j];
                break;
            case 10:
            case 12:
            case 16:
                r_y = ((uint16_t*)ref->data[0])[i * (ref->stride[0] / 2) + j];
                r_u = ((uint16_t*)ref->data[1])[i * (ref->stride[1] / 2) + j];
                r_v = ((uint16_t*)ref->data[2])[i * (ref->stride[2] / 2) + j];
                d_y = ((uint16_t*)dist->data[0])[i * (dist->stride[0] / 2) + j];
                d_u = ((uint16_t*)dist->data[1])[i * (dist->stride[1] / 2) + j];
                d_v = ((uint16_t*)dist->data[2])[i * (dist->stride[2] / 2) + j];
                break;
            default:
                return -EINVAL;
            }

            const LABColor color_1 = get_lab_color(r_y, r_u, r_v, ref->bpc);
            const LABColor color_2 = get_lab_color(d_y, d_u, d_v, dist->bpc);
            const KSubArgs default_ksub = { .l = 0.65, .c = 1.0, .h = 4.0 };
            const float de00 = ciede2000(color_1, color_2, default_ksub);
            de00_sum += de00;
        }
    }

    const double score = 45. - 20. *
                         log10(de00_sum / (ref_pic->w[0] * ref_pic->h[0]));
    return vmaf_feature_collector_append(feature_collector, "ciede2000", score,
                                         index);
}

static int close(VmafFeatureExtractor *fex)
{
    CiedeState *s = fex->priv;
    if (s->ref.data[0] && s->dist.data[0]) {
        vmaf_picture_unref(&s->ref);
        vmaf_picture_unref(&s->dist);
    }
    return 0;
}

static const char *provided_features[] = {
    "ciede2000",
    NULL
};

VmafFeatureExtractor vmaf_fex_ciede = {
    .name = "ciede",
    .init = init,
    .extract = extract,
    .close = close,
    .priv_size = sizeof(CiedeState),
    .provided_features = provided_features,
};
