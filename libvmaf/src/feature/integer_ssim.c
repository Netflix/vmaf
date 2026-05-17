/*
Copyright 2001-2012 Xiph.Org and contributors.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"
#include "opt.h"

#define KERNEL_SHIFT (8)
#define KERNEL_WEIGHT (1 << KERNEL_SHIFT)
#define KERNEL_ROUND ((1 << KERNEL_SHIFT) >> 1)

#ifndef M_PI
#define M_PI 3.141592653589793238462643
#endif

static int gaussian_filter_init(unsigned **iqa_kernel, double _sigma, int _max_len)
{
    unsigned *kernel;
    double scale;
    double nhisigma2;
    double s;
    double len;
    unsigned sum;
    int kernel_len;
    int kernel_sz;
    int ci;
    scale = 1 / (sqrt(2 * M_PI) * _sigma);
    nhisigma2 = -0.5 / (_sigma * _sigma);
    /*Compute the kernel size so that the error in the first truncated
     coefficient is no larger than 0.5*KERNEL_WEIGHT.
    There is no point in going beyond this given our working precision.*/
    s = sqrt(0.5 * M_PI) * _sigma * (1.0 / KERNEL_WEIGHT);
    if (s >= 1) {
        len = 0;
    } else {
        len = floor(_sigma * sqrt(-2 * log(s)));
    }
    kernel_len = len >= _max_len ? _max_len - 1 : (int)len;
    kernel_sz = kernel_len << 1 | 1;
    kernel = (unsigned *)malloc(kernel_sz * sizeof(*kernel));
    sum = 0;
    for (ci = kernel_len; ci > 0; ci--) {
        kernel[kernel_len - ci] = kernel[kernel_len + ci] =
            (unsigned)(KERNEL_WEIGHT * scale * exp(nhisigma2 * ci * ci) + 0.5);
        sum += kernel[kernel_len - ci];
    }
    kernel[kernel_len] = KERNEL_WEIGHT - (sum << 1);
    *iqa_kernel = kernel;
    return kernel_sz;
}

typedef struct ssim_moments ssim_moments;

struct ssim_moments {
    int64_t mux;
    int64_t muy;
    int64_t x2;
    int64_t xy;
    int64_t y2;
    int64_t w;
};

#define SSIM_K1 (0.01 * 0.01)
#define SSIM_K2 (0.03 * 0.03)

/* Compute a power-of-two ring-buffer size >= kernel size. */
static int ssim_line_buffer_size(int vkernel_sz)
{
    int line_sz;
    int log_line_sz;
    for (line_sz = 1, log_line_sz = 0; line_sz < vkernel_sz; line_sz <<= 1, log_line_sz++)
        ;
    (void)log_line_sz;
    return line_sz;
}

/* Horizontal 1-D accumulation of moments for a single row into `buf`. */
static void ssim_accumulate_row(const unsigned char *src, const unsigned char *dst, int w,
                                int depth, const unsigned *hkernel, int hkernel_sz,
                                int hkernel_offs, ssim_moments *buf)
{
    for (int x = 0; x < w; x++) {
        ssim_moments m;
        int k;
        int k_min;
        int k_max;
        memset(&m, 0, sizeof(m));
        k_min = hkernel_offs - x <= 0 ? 0 : hkernel_offs - x;
        k_max =
            x + hkernel_offs - w + 1 <= 0 ? hkernel_sz : hkernel_sz - (x + hkernel_offs - w + 1);
        // k_min/k_max clamp to in-bounds — analyzer can't prove kernel
        // offsets stay within hkernel[0..hkernel_sz) and src/dst[0.._w) here.
        // NOLINTBEGIN(clang-analyzer-security.ArrayBound)
        for (k = k_min; k < k_max; k++) {
            signed s;
            signed d;
            signed window;
            const ptrdiff_t off = (ptrdiff_t)x - (ptrdiff_t)hkernel_offs + (ptrdiff_t)k;
            if (depth > 8) {
                s = src[off * 2] + (src[off * 2 + 1] << 8);
                d = dst[off * 2] + (dst[off * 2 + 1] << 8);
            } else {
                s = src[off];
                d = dst[off];
            }
            window = hkernel[k];
            m.mux += (int64_t)window * s;
            m.muy += (int64_t)window * d;
            m.x2 += (int64_t)window * s * s;
            m.xy += (int64_t)window * s * d;
            m.y2 += (int64_t)window * d * d;
            m.w += window;
        }
        // NOLINTEND(clang-analyzer-security.ArrayBound)
        buf[x] = m;
    }
}

/* Vertical 1-D accumulation + SSIM term for one output row. */
static void ssim_reduce_row_range(ssim_moments *const *lines, int line_mask, int y, int w,
                                  int vkernel_sz, const unsigned *vkernel, int samplemax, int k_min,
                                  int k_max, double *ssim, double *ssimw)
{
    for (int x = 0; x < w; x++) {
        ssim_moments m;
        const ssim_moments *buf;
        double c1;
        double c2;
        double mx2;
        double mxy;
        double my2;
        double w_d;
        int k;
        memset(&m, 0, sizeof(m));
        // k_min/k_max clamp to in-bounds — analyzer can't prove kernel
        // offsets stay within vkernel[0..vkernel_sz) here.
        // NOLINTBEGIN(clang-analyzer-security.ArrayBound)
        for (k = k_min; k < k_max; k++) {
            signed window;
            buf = lines[(y + 1 - vkernel_sz + k) & line_mask] + x;
            window = vkernel[k];
            m.mux += window * buf->mux;
            m.muy += window * buf->muy;
            m.x2 += window * buf->x2;
            m.xy += window * buf->xy;
            m.y2 += window * buf->y2;
            m.w += window * buf->w;
        }
        // NOLINTEND(clang-analyzer-security.ArrayBound)
        w_d = m.w;
        c1 = samplemax * samplemax * SSIM_K1 * w_d * w_d;
        c2 = samplemax * samplemax * SSIM_K2 * w_d * w_d;
        mx2 = m.mux * (double)m.mux;
        mxy = m.mux * (double)m.muy;
        my2 = m.muy * (double)m.muy;
        *ssim += m.w * (2 * mxy + c1) * (c2 + 2 * (m.xy * w_d - mxy)) /
                 ((mx2 + my2 + c1) * (m.x2 * w_d - mx2 + m.y2 * w_d - my2 + c2));
        *ssimw += m.w;
    }
}

static double calc_ssim(const unsigned char *_src, int _systride, const unsigned char *_dst,
                        int _dystride, double _par, int depth, int _w, int _h)
{
    (void)_par;
    ssim_moments *line_buf;
    ssim_moments **lines;
    double ssim;
    double ssimw;
    unsigned *hkernel;
    int hkernel_sz;
    int hkernel_offs;
    unsigned *vkernel;
    int vkernel_sz;
    int vkernel_offs;
    int line_sz;
    int line_mask;
    int y;
    int samplemax;
    samplemax = (1 << depth) - 1;
    vkernel_sz = gaussian_filter_init(&vkernel, 1.5, 5);
    vkernel_offs = vkernel_sz >> 1;
    line_sz = ssim_line_buffer_size(vkernel_sz);
    line_mask = line_sz - 1;
    lines = (ssim_moments **)malloc((size_t)line_sz * sizeof(*lines));
    lines[0] = line_buf = (ssim_moments *)malloc((size_t)line_sz * (size_t)_w * sizeof(*line_buf));
    for (y = 1; y < line_sz; y++)
        lines[y] = lines[y - 1] + _w;
    hkernel_sz = gaussian_filter_init(&hkernel, 1.5, 5);
    hkernel_offs = hkernel_sz >> 1;
    ssim = 0;
    ssimw = 0;
    for (y = 0; y < _h + vkernel_offs; y++) {
        if (y < _h) {
            ssim_accumulate_row(_src, _dst, _w, depth, hkernel, hkernel_sz, hkernel_offs,
                                lines[y & line_mask]);
            _src += _systride;
            _dst += _dystride;
        }
        if (y >= vkernel_offs) {
            int k_min = vkernel_sz - y - 1 <= 0 ? 0 : vkernel_sz - y - 1;
            int k_max = y + 1 - _h <= 0 ? vkernel_sz : vkernel_sz - (y + 1 - _h);
            ssim_reduce_row_range(lines, line_mask, y, _w, vkernel_sz, vkernel, samplemax, k_min,
                                  k_max, &ssim, &ssimw);
        }
    }
    free(line_buf);
    free((void *)lines);
    free(vkernel);
    free(hkernel);
    return ssim / ssimw;
}

typedef struct IntegerSsimState {
    bool enable_db;
    bool clip_db;
    double max_db;
} IntegerSsimState;

static const VmafOption options[] = {{
                                         .name = "enable_db",
                                         .help = "write SSIM values as dB: -10*log10(1-ssim)",
                                         .offset = offsetof(IntegerSsimState, enable_db),
                                         .type = VMAF_OPT_TYPE_BOOL,
                                         .default_val.b = false,
                                     },
                                     {
                                         .name = "clip_db",
                                         .help = "clip dB scores to a peak-derived ceiling",
                                         .offset = offsetof(IntegerSsimState, clip_db),
                                         .type = VMAF_OPT_TYPE_BOOL,
                                         .default_val.b = false,
                                     },
                                     {0}};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                unsigned h)
{
    (void)pix_fmt;

    IntegerSsimState *s = fex->priv;

    const unsigned peak = (1u << bpc) - 1u;
    if (s->clip_db) {
        const double mse = 0.5 / ((double)w * (double)h);
        s->max_db = ceil(10. * log10((double)(peak * peak) / mse));
    } else {
        s->max_db = INFINITY;
    }

    return 0;
}

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

static int extract(VmafFeatureExtractor *fex, VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90, unsigned index,
                   VmafFeatureCollector *feature_collector)
{
    IntegerSsimState *s = fex->priv;
    (void)ref_pic_90;
    (void)dist_pic_90;

    double score = calc_ssim(ref_pic->data[0], ref_pic->stride[0], dist_pic->data[0],
                             dist_pic->stride[0], 1.0, ref_pic->bpc, ref_pic->w[0], ref_pic->h[0]);

    if (s->enable_db)
        score = MIN(-10. * log10(1. - score), s->max_db);

    int err = vmaf_feature_collector_append(feature_collector, "ssim", score, index);
    if (err)
        return err;
    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    (void)fex;
    return 0;
}

static const char *provided_features[] = {"ssim", NULL};

/* Fixed-point SSIM extractor — registered as `ssim` in
 * `feature_extractor.c`'s `feature_extractor_list[]`. The companion
 * `float_ssim.c` provides the floating-point variant under the
 * `float_ssim` name. The cross-TU reference from
 * `feature_extractor.c` is invisible to clang-tidy's per-TU
 * analysis, so the linkage check fires a false positive here. */
// NOLINTNEXTLINE(misc-use-internal-linkage)
VmafFeatureExtractor vmaf_fex_ssim = {
    .name = "ssim",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(IntegerSsimState),
    .provided_features = provided_features,
};
