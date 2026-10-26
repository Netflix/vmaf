/**
 *
 *  Copyright 2026 luoye-cpu
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

/*
 * SpEED chroma device code (speed_chroma_cuda).
 *
 * IMPORTANT: this translation unit is *only* compiled to a fatbin by nvcc and
 * then embedded as a byte array by bin2c (see src/meson.build, cuda_cu_sources
 * / ptx_bin2c_*). Host code placed here is never compiled or linked. All host
 * side orchestration lives in feature/cuda/integer_speed_chroma_cuda.c, which
 * loads the `speed_chroma_ptx` module through cuModuleLoadData().
 *
 * Numerical parity with feature/speed.c (the CPU reference) is the contract:
 *
 *  - Every kernel mirrors the loop structure, the accumulation order and the
 *    operand types (float vs double) of its CPU counterpart. No parallel
 *    reduction is used anywhere: each output element/tap is accumulated
 *    sequentially in exactly the CPU's order, on a single thread.
 *
 *  - nvcc contracts `a * b + c` into an FMA by default, which changes the
 *    rounding of the result. Every arithmetic operation that feeds an
 *    accumulation is therefore written with the rounding intrinsics
 *    (__fadd_rn/__fmul_rn/__fsub_rn/__fdiv_rn/__fsqrt_rn and the double
 *    versions); those are never fused by the compiler. This file must also
 *    never be compiled with --use_fast_math (see src/meson.build).
 *
 *  - The CPU accidentally promotes some expressions to double through the
 *    double-typed libm entry points (sqrt/log2 take/return double) and through
 *    double literals, e.g. `-0.5 * tau * xv`, `1 + t * t` inside `1.0 /
 *    sqrt(...)`, or `(ta - tb) / 2.0`. Those are reproduced here as double
 *    operations followed by the identical (implicit) rounding back to float.
 *
 *  - Functions the CPU computes on the host with the platform libm (the
 *    gaussian filter taps, `log2` in the entropy update and in
 *    get_speed_score) are deliberately *not* ported here: they live in the
 *    host wrapper, where the very same libm is used, so the results are
 *    identical by construction instead of merely close.
 */

#include "cuda_helper.cuh"

#define SPEED_BLOCK_SIZE (5)
#define SPEED_ELEMENTS_IN_BLOCK (25)
#define SPEED_QR_SIZE (25)
#define EIGENVALUE_EPS (1e-6)
#define EIGENVALUE_MAX_ITERS (500)

/* Scratch layout used by compute_eigenvalues() (size == 25):
 * A (size*size) + d (size) + sd (size) + tmp (2*size). */
#define SPEED_EIGEN_SHARED (SPEED_QR_SIZE * SPEED_QR_SIZE + 4 * SPEED_QR_SIZE)

/* Rounding-exact arithmetic. The CPU build (MSVC, /fp:precise) is scalar and
 * never contracts a multiply with an add; the intrinsics below guarantee the
 * device does the same. */
#define FADD(a, b) __fadd_rn(a, b)
#define FSUB(a, b) __fsub_rn(a, b)
#define FMUL(a, b) __fmul_rn(a, b)
#define FDIV(a, b) __fdiv_rn(a, b)
#define FSQRT(a) __fsqrt_rn(a)
#define DADD(a, b) __dadd_rn(a, b)
#define DSUB(a, b) __dsub_rn(a, b)
#define DMUL(a, b) __dmul_rn(a, b)
#define DDIV(a, b) __ddiv_rn(a, b)
#define DSQRT(a) __dsqrt_rn(a)

extern "C" {

/* ------------------------------------------------------------------------- */
/* Floating point helpers                                                     */
/* ------------------------------------------------------------------------- */

/* vector_norm()/compute_column_norm()/pythagoras() in the CPU all end in
 * `sqrt(float expression)`: the argument is promoted to double, the double
 * sqrt is taken and the result is rounded back to float on assignment. */
__device__ __forceinline__ float speed_sqrtd_roundf(float x)
{
    return (float)DSQRT((double)x);
}

__device__ __forceinline__ int speed_get_sign(float x)
{
    return (x >= 0 ? 1 : -1);
}

__device__ __forceinline__ float speed_pythagoras(float x, float y)
{
    return speed_sqrtd_roundf(FADD(FMUL(x, x), FMUL(y, y)));
}

/* ------------------------------------------------------------------------- */
/* Picture conversion (picture_copy)                                          */
/* ------------------------------------------------------------------------- */

/* picture_copy(): dst = (float)src + offset for 8-bit input. The CPU treats
 * every bitdepth other than 10/12/16 as 8-bit. */
__global__ void speed_chroma_copy8_kernel(const uint8_t *src, int src_stride,
                                          float *dst, int dst_stride,
                                          int w, int h, float offset)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= h || j >= w) return;

    const float v = (float)src[(size_t)i * src_stride + j];
    dst[(size_t)i * dst_stride + j] = FADD(v, offset);
}

/* picture_copy_hbd(): dst = (float)src / scaler + offset. scaler is 4/16/256,
 * so the division is exact. */
__global__ void speed_chroma_copy16_kernel(const uint16_t *src, int src_stride,
                                           float *dst, int dst_stride,
                                           int w, int h, float scaler,
                                           float offset)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= h || j >= w) return;

    const float v = (float)src[(size_t)i * src_stride + j];
    dst[(size_t)i * dst_stride + j] = FADD(FDIV(v, scaler), offset);
}

/* ------------------------------------------------------------------------- */
/* vif_scale_frame_s() - the prescale step of filter_and_downscale()          */
/* ------------------------------------------------------------------------- */

/* vif_tools.c mirror(): `i` is a float (the floor()/ceil() results are
 * converted back to float on the call), the bounds are ints converted to
 * float; the result is assigned to an int, i.e. truncated. */
__device__ __forceinline__ float speed_scale_mirror(float i, float left,
                                                    float right)
{
    return (i < left ? -i : i > right ? 2 * right - i : i);
}

/* bilinear_interpolation() in vif_tools.c: the four taps are accumulated
 * left-to-right, each term a float multiply of two float factors. */
__device__ __forceinline__ float speed_scale_bilinear_interpolation(
        const float *src, int width, int height, int src_stride, float x, float y)
{
    const float xf = (float)floor((double)x);
    const float xc = (float)ceil((double)x);
    const float yf = (float)floor((double)y);
    const float yc = (float)ceil((double)y);

    const int x1 = (int)speed_scale_mirror(xf, 0.0f, (float)(width - 1));
    const int x2 = (int)speed_scale_mirror(xc, 0.0f, (float)(width - 1));
    const int y1 = (int)speed_scale_mirror(yf, 0.0f, (float)(height - 1));
    const int y2 = (int)speed_scale_mirror(yc, 0.0f, (float)(height - 1));

    const float dx = FSUB(x, (float)x1);
    const float dy = FSUB(y, (float)y1);
    const float idx = FSUB(1.0f, dx);
    const float idy = FSUB(1.0f, dy);

    const float v11 = src[(size_t)y1 * src_stride + x1];
    const float v12 = src[(size_t)y1 * src_stride + x2];
    const float v21 = src[(size_t)y2 * src_stride + x1];
    const float v22 = src[(size_t)y2 * src_stride + x2];

    return FADD(FADD(FADD(FMUL(FMUL(idy, idx), v11), FMUL(FMUL(idy, dx), v12)),
                     FMUL(FMUL(dy, idx), v21)),
                FMUL(FMUL(dy, dx), v22));
}

/* vif_scale_frame_nearest_s(): dst[y][x] = src[(int)(y * ratio_y)][(int)(x * ratio_x)]. */
__global__ void speed_chroma_scale_nearest_kernel(const float *src, float *dst,
                                                  int src_w, int src_h,
                                                  int src_stride, int dst_w,
                                                  int dst_h, int dst_stride)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= dst_h || j >= dst_w) return;

    const float ratio_x = FDIV((float)src_w, (float)dst_w);
    const float ratio_y = FDIV((float)src_h, (float)dst_h);

    const int rounded_y = (int)FMUL((float)i, ratio_y);
    const int rounded_x = (int)FMUL((float)j, ratio_x);

    dst[(size_t)i * dst_stride + j] =
        src[(size_t)rounded_y * src_stride + rounded_x];
}

/* vif_scale_frame_bilinear_s(). float xx = (x + 0.5) * ratio_x - 0.5; is
 * evaluated in double on the CPU (0.5 is a double literal) and rounded once
 * when it is assigned to the float. */
__global__ void speed_chroma_scale_bilinear_kernel(const float *src, float *dst,
                                                   int src_w, int src_h,
                                                   int src_stride, int dst_w,
                                                   int dst_h, int dst_stride)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= dst_h || j >= dst_w) return;

    const float ratio_x = FDIV((float)src_w, (float)dst_w);
    const float ratio_y = FDIV((float)src_h, (float)dst_h);

    const float xx = (float)DSUB(DMUL(DADD((double)j, 0.5), (double)ratio_x),
                                 0.5);
    const float yy = (float)DSUB(DMUL(DADD((double)i, 0.5), (double)ratio_y),
                                 0.5);

    dst[(size_t)i * dst_stride + j] =
        speed_scale_bilinear_interpolation(src, src_w, src_h, src_stride, xx, yy);
}

/* ------------------------------------------------------------------------- */
/* filter_and_downscale helpers (vif_tools.c)                                 */
/* ------------------------------------------------------------------------- */

/*
 * vif_filter1d_s(), pass 1: vertical convolution with edge reflection.
 * The CPU computes tmp[] for one output row and immediately consumes it in the
 * horizontal pass, so materialising all rows of tmp up front is equivalent:
 * every row is fully written before it is read.
 */
__global__ void speed_chroma_filter1d_vert_kernel(const float *f, const float *src,
                                                  float *tmp, int w, int h,
                                                  int src_stride, int fwidth)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= h || j >= w) return;

    float accum = 0.0f;
    for (int fi = 0; fi < fwidth; ++fi) {
        const float fcoeff = f[fi];
        int ii = i - fwidth / 2 + fi;
        ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 2 : ii);
        const float imgcoeff = src[(size_t)ii * src_stride + j];
        accum = FADD(accum, FMUL(fcoeff, imgcoeff));
    }
    tmp[(size_t)i * w + j] = accum;
}

/* vif_filter1d_s(), pass 2: horizontal convolution on the row-major tmp buffer
 * (tmp has row stride w, exactly like the CPU's per-row scratch). */
__global__ void speed_chroma_filter1d_horiz_kernel(const float *f, const float *tmp,
                                                   float *dst, int w, int h,
                                                   int dst_stride, int fwidth)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= h || j >= w) return;

    float accum = 0.0f;
    for (int fj = 0; fj < fwidth; ++fj) {
        const float fcoeff = f[fj];
        int jj = j - fwidth / 2 + fj;
        jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 2 : jj);
        const float imgcoeff = tmp[(size_t)i * w + jj];
        accum = FADD(accum, FMUL(fcoeff, imgcoeff));
    }
    dst[(size_t)i * dst_stride + j] = accum;
}

/* vif_dec16_s(): decimation by 16. */
__global__ void speed_chroma_dec16_kernel(const float *src, float *dst,
                                          int src_w, int src_h, int src_stride,
                                          int dst_stride)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= src_h / 16 || j >= src_w / 16) return;

    dst[(size_t)i * dst_stride + j] = src[(size_t)(i * 16) * src_stride + (j * 16)];
}

/* subtract_image(): im1 -= im2 on the top-left w x h window. */
__global__ void speed_chroma_subtract_kernel(float *im1, const float *im2,
                                             int w, int h, int stride)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= h || j >= w) return;

    const size_t idx = (size_t)i * stride + j;
    im1[idx] = FSUB(im1[idx], im2[idx]);
}

/* ------------------------------------------------------------------------- */
/* est_params, step 1: covariance matrix (compute_covariance_matrix)          */
/* ------------------------------------------------------------------------- */

/* compute_mean(): float accumulation over the (submatrix_height,
 * submatrix_width) window, in row-major order. */
__global__ void speed_chroma_means_kernel(const float *data, float *means,
                                          int stride, int block_size,
                                          int sub_w, int sub_h)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= block_size * block_size) return;

    const int start_row = idx / block_size;
    const int start_col = idx % block_size;

    float result = 0.0f;
    for (int i = 0; i < sub_h; i++) {
        for (int j = 0; j < sub_w; j++)
            result = FADD(result, data[(size_t)(start_row + i) * stride +
                                       (start_col + j)]);
    }

    /* `result / (submatrix_width * submatrix_height)` with both sizes promoted
     * to float. sub_w * sub_h == 0 gives NaN, exactly like the CPU. */
    means[idx] = FDIV(result, (float)(sub_w * sub_h));
}

/*
 * compute_covariance_matrix(): one thread per (x_index, y_index) pair with
 * y_index <= x_index (325 threads). The covariance sum runs in double, in the
 * same i,j order as compute_cov_kernel_scalar(), and the /N division is done
 * in double before the float store.
 */
__global__ void speed_chroma_cov_kernel(const float *data, const float *means,
                                        float *cov_mat, int stride,
                                        int block_size, int elements_in_block,
                                        int sub_w, int sub_h)
{
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= elements_in_block * (elements_in_block + 1) / 2) return;

    /* decode pair -> (x_index, y_index) with y_index <= x_index */
    int x_index = 0, base = 0;
    while (base + x_index + 1 <= pair) {
        base += x_index + 1;
        x_index++;
    }
    const int y_index = pair - base;

    const int start_row_x = x_index / block_size;
    const int start_col_x = x_index % block_size;
    const int start_row_y = y_index / block_size;
    const int start_col_y = y_index % block_size;

    const double mean_x = (double)means[start_row_x * block_size + start_col_x];
    const double mean_y = (double)means[start_row_y * block_size + start_col_y];

    const float *data_x = data + (size_t)start_row_x * stride + start_col_x;
    const float *data_y = data + (size_t)start_row_y * stride + start_col_y;

    double result = 0.0;
    for (int i = 0; i < sub_h; i++) {
        for (int j = 0; j < sub_w; j++) {
            const double val_x = (double)data_x[(size_t)i * stride + j];
            const double val_y = (double)data_y[(size_t)i * stride + j];
            result = DADD(result, DMUL(DSUB(val_x, mean_x), DSUB(val_y, mean_y)));
        }
    }

    const float covariance = (float)DDIV(result, (double)(sub_w * sub_h));
    cov_mat[x_index * elements_in_block + y_index] = covariance;
    cov_mat[y_index * elements_in_block + x_index] = covariance;
}

/* ------------------------------------------------------------------------- */
/* est_params, step 2: eigenvalues (Householder + implicit QL, gsl port)      */
/* ------------------------------------------------------------------------- */

__device__ float speed_compute_column_norm(const float *A, int col, int start_row,
                                           int size)
{
    float norm = 0.0f;
    for (int i = start_row; i < size; i++)
        norm = FADD(norm, FMUL(A[i * size + col], A[i * size + col]));
    return speed_sqrtd_roundf(norm);
}

__device__ float speed_compute_householder_transform(float *A, int col,
                                                     int start_row, int size)
{
    if (size - start_row == 1)
        return 0.0f;
    const float xnorm = speed_compute_column_norm(A, col, start_row + 1, size);
    if (xnorm == 0.0f)
        return 0.0f;
    const float alpha = A[start_row * size + col];
    const float pyth = speed_pythagoras(alpha, xnorm);
    /* -get_sign(alpha) * pyth: an int times a float, i.e. an exact sign flip */
    const float beta = FMUL((float)(-speed_get_sign(alpha)), pyth);
    const float tau = FDIV(FSUB(beta, alpha), beta);
    const float s = FSUB(alpha, beta);
    if (s != 0.0f) {
        for (int i = start_row; i < size; i++)
            A[i * size + col] = FDIV(A[i * size + col], s);
        A[start_row * size + col] = beta;
    }
    return tau;
}

__device__ void speed_tridiagonal_multiply(float *A, float *v, float *x,
                                           float tau_i, int start, int size)
{
    for (int i = start; i < size; i++) {
        x[i] = 0.0f;
        for (int j = start; j < size; j++)
            x[i] = FADD(x[i], FMUL(FMUL(tau_i, A[i * size + j]), v[j]));
    }
}

__device__ float speed_tridiagonal_dot_product(float *x, float *v, int start,
                                               int size)
{
    float res = 0.0f;
    for (int i = start; i < size; i++)
        res = FADD(res, FMUL(x[i], v[i]));
    return res;
}

__device__ void speed_tridiagonal_axpy(float *x, float *v, float alpha,
                                       int start, int size)
{
    for (int i = start; i < size; i++)
        x[i] = FADD(x[i], FMUL(alpha, v[i]));
}

__device__ void speed_tridiagonal_syr2(float *A, float *x, float *v, int start,
                                       int size)
{
    for (int i = start; i < size; i++) {
        for (int j = start; j < size; j++)
            A[i * size + j] = FSUB(A[i * size + j],
                                   FADD(FMUL(x[i], v[j]), FMUL(v[i], x[j])));
    }
}

/* convert_to_tridiagonal(): adapted gsl_linalg_symmtd_decomp. */
__device__ void speed_convert_to_tridiagonal(float *A, int size, float *d,
                                             float *sd, float *buffer)
{
    float *v = buffer;
    float *x = buffer + size;

    for (int i = 0; i < size - 2; i++) {
        const float tau_i = speed_compute_householder_transform(A, i, i + 1, size);

        for (int j = i + 1; j < size; j++)
            v[j] = A[j * size + i];

        if (tau_i != 0.0f) {
            A[(i + 1) * size + i] = v[i + 1];
            v[i + 1] = 1.0f;
            speed_tridiagonal_multiply(A, v, x, tau_i, i + 1, size);
            const float xv = speed_tridiagonal_dot_product(x, v, i + 1, size);
            /* the CPU computes `-0.5 * tau_i * xv` in double */
            const float alpha = (float)DMUL(DMUL(-0.5, (double)tau_i), (double)xv);
            speed_tridiagonal_axpy(x, v, alpha, i + 1, size);
            speed_tridiagonal_syr2(A, x, v, i + 1, size);
        }
    }

    for (int i = 0; i < size; i++)
        d[i] = A[i * size + i];
    for (int i = 0; i < size - 1; i++)
        sd[i] = A[(i + 1) * size + i];
}

__device__ void speed_chop_small_elements(float *d, float *sd, int size)
{
    for (int i = 0; i < size - 1; i++) {
        /* EIGENVALUE_EPS is a double literal: the comparison happens in double,
         * with fabsf(d[i]) + fabsf(d[i+1]) rounded to float first. */
        if ((double)fabsf(sd[i]) <
            EIGENVALUE_EPS * (double)FADD(fabsf(d[i]), fabsf(d[i + 1])))
            sd[i] = 0.0f;
    }
}

__device__ float speed_trailing_eigenvalue(float *d, float *sd, int n)
{
    const float ta = d[n - 2];
    const float tb = d[n - 1];
    const float tab = sd[n - 2];
    const float dt = (float)DDIV((double)FSUB(ta, tb), 2.0);

    if (dt > 0)
        return FSUB(tb, FMUL(tab, FDIV(tab, FADD(dt, speed_pythagoras(dt, tab)))));
    else if (dt == 0)
        return FSUB(tb, fabsf(tab));
    else
        return FADD(tb, FMUL(tab, FDIV(tab, FADD(-dt, speed_pythagoras(dt, tab)))));
}

/* create_givens(): `1.0 / sqrt(1 + t * t)` is a *double* reciprocal square
 * root because of the double literals, rounded to float on assignment. */
__device__ void speed_create_givens(const float a, const float b, float *c,
                                    float *s)
{
    if (b == 0) {
        *c = 1;
        *s = 0;
    } else if (fabsf(b) > fabsf(a)) {
        const float t = FDIV(-a, b);
        const float s1 = (float)DDIV(1.0, DSQRT((double)FADD(1.0f, FMUL(t, t))));
        *s = s1;
        *c = FMUL(s1, t);
    } else {
        const float t = FDIV(-b, a);
        const float c1 = (float)DDIV(1.0, DSQRT((double)FADD(1.0f, FMUL(t, t))));
        *c = c1;
        *s = FMUL(c1, t);
    }
}

__device__ void speed_qr_step(float *d, float *sd, int n)
{
    float mu = speed_trailing_eigenvalue(d, sd, n);
    if (EIGENVALUE_EPS * (double)fabsf(mu) >
        (double)FADD(fabsf(d[0]), fabsf(sd[0])))
        mu = 0;

    float x = FSUB(d[0], mu);
    float z = sd[0];

    float ak = 0;
    float bk = 0;
    float zk = 0;

    float ap = d[0];
    float bp = sd[0];

    float aq = d[1];

    if (n == 2) {
        float c, s;
        speed_create_givens(x, z, &c, &s);

        const float ap1 = FADD(FMUL(c, FSUB(FMUL(c, ap), FMUL(s, bp))),
                               FMUL(s, FSUB(FMUL(s, aq), FMUL(c, bp))));
        const float bp1 = FSUB(FMUL(c, FADD(FMUL(s, ap), FMUL(c, bp))),
                               FMUL(s, FADD(FMUL(s, bp), FMUL(c, aq))));
        const float aq1 = FADD(FMUL(s, FADD(FMUL(s, ap), FMUL(c, bp))),
                               FMUL(c, FADD(FMUL(s, bp), FMUL(c, aq))));

        ak = ap1;
        bk = bp1;

        ap = aq1;

        d[0] = ak;
        sd[0] = bk;
        d[1] = ap;

        return;
    }

    float bq = sd[1];

    for (int k = 0; k < n - 1; k++) {
        float c, s;
        speed_create_givens(x, z, &c, &s);

        /* compute G' T G */
        const float bk1 = FSUB(FMUL(c, bk), FMUL(s, zk));

        const float ap1 = FADD(FMUL(c, FSUB(FMUL(c, ap), FMUL(s, bp))),
                               FMUL(s, FSUB(FMUL(s, aq), FMUL(c, bp))));
        const float bp1 = FSUB(FMUL(c, FADD(FMUL(s, ap), FMUL(c, bp))),
                               FMUL(s, FADD(FMUL(s, bp), FMUL(c, aq))));
        const float zp1 = FMUL(-s, bq);

        const float aq1 = FADD(FMUL(s, FADD(FMUL(s, ap), FMUL(c, bp))),
                               FMUL(c, FADD(FMUL(s, bp), FMUL(c, aq))));
        const float bq1 = FMUL(c, bq);

        ak = ap1;
        bk = bp1;
        zk = zp1;

        ap = aq1;
        bp = bq1;

        if (k < n - 2)
            aq = d[k + 2];

        if (k < n - 3)
            bq = sd[k + 2];

        d[k] = ak;

        if (k > 0)
            sd[k - 1] = bk1;

        if (k < n - 2)
            sd[k + 1] = bp;

        x = bk;
        z = zk;
    }

    d[n - 1] = ap;
    sd[n - 2] = bk;
}

/* compute_eigenvalues_tridiagonal(): adapted gsl_eigen_symm. The CPU logs a
 * warning when EIGENVALUE_MAX_ITERS is reached; that log has no effect on the
 * eigenvalues and is intentionally not reproduced here. */
__device__ void speed_compute_eigenvalues_tridiagonal(float *d, float *sd,
                                                      float *eigenvalues, int size)
{
    speed_chop_small_elements(d, sd, size);

    int b = size - 1;
    int iter = 0;
    while (b > 0 && iter < EIGENVALUE_MAX_ITERS) {
        if (sd[b - 1] == 0.0f) {
            b--;
            continue;
        }
        int a = b - 1;
        while (a > 0) {
            if (sd[a - 1] == 0.0f)
                break;
            a--;
        }

        const int n_block = b - a + 1;
        float *d_block = d + a;
        float *sd_block = sd + a;
        speed_qr_step(d_block, sd_block, n_block);
        speed_chop_small_elements(d_block, sd_block, n_block);
        iter++;
    }

    for (int i = 0; i < size; i++)
        eigenvalues[i] = d[i];
}

/*
 * compute_eigenvalues(): single thread -- the whole decomposition is
 * sequential in the CPU and only runs once per image.
 */
__global__ void speed_chroma_eigen_kernel(const float *A_in, float *eigenvalues,
                                          int size)
{
    __shared__ float sbuf[SPEED_EIGEN_SHARED];
    float *A = sbuf;
    float *d = sbuf + SPEED_QR_SIZE * SPEED_QR_SIZE;
    float *sd = d + SPEED_QR_SIZE;
    float *tmp = sd + SPEED_QR_SIZE;

    for (int i = 0; i < size * size; i++)
        A[i] = A_in[i];

    if (size == 1) {
        eigenvalues[0] = A[0 * size + 0];
        return;
    }

    speed_convert_to_tridiagonal(A, size, d, sd, tmp);
    speed_compute_eigenvalues_tridiagonal(d, sd, eigenvalues, size);
}

/* ------------------------------------------------------------------------- */
/* est_params, step 3: independent term (compute_independent_term)           */
/* ------------------------------------------------------------------------- */

__global__ void speed_chroma_independent_term_kernel(
    const float *data, float *independent_term, int stride, int block_size,
    int truncated_width, int truncated_height, int num_blocks_horizontal,
    int num_blocks)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= block_size * block_size) return;

    const int start_i = idx / block_size;
    const int start_j = idx % block_size;

    for (int i = start_i; i < truncated_height; i += block_size) {
        for (int j = start_j; j < truncated_width; j += block_size) {
            const int out_row = start_i * block_size + start_j;
            const int out_col = ((i - start_i) / block_size) * num_blocks_horizontal
                              + ((j - start_j) / block_size);
            independent_term[(size_t)out_row * num_blocks + out_col] =
                data[(size_t)i * stride + j];
        }
    }
}

/* ------------------------------------------------------------------------- */
/* est_params, step 4: linear system (matrix_qr_decomposition + solve)        */
/* ------------------------------------------------------------------------- */

/*
 * matrix_qr_decomposition() + the two matrix_mul() calls of
 * solve_linear_system(): R = Q * A and tmp_rect = Q * B.
 *
 * The CPU transposes Q inside matrix_qr_decomposition() and transposes it back
 * in solve_linear_system(), i.e. the two transposes cancel and both products
 * use the accumulated factor Q as-is; that is what is computed here.
 *
 * The whole decomposition is a chain of 24 steps, each step being a
 * Householder reflection followed by two matrix products. Parallelising the
 * elementwise work is bit-exact (each output element still accumulates its
 * products in the CPU's k order on one thread); the sequential parts (the
 * norms and the normalisation of v) are done by thread 0. The error flags
 * (irregular eigenvalues / singular triangular factor), the zcopy of the
 * solution and the entropy update depend on this state, so they are produced
 * here or in the kernels below, never on the host.
 */
__global__ void speed_chroma_qr_kernel(const float *A_in, const float *eigenvalues,
                                       const float *B_in, float *R_out,
                                       float *TmpRect_out, float *X_out,
                                       int num_blocks, int *err_flag)
{
    const int size = SPEED_QR_SIZE;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int total = size * num_blocks;

    __shared__ float sA[SPEED_QR_SIZE * SPEED_QR_SIZE];
    __shared__ float sTz[2][SPEED_QR_SIZE * SPEED_QR_SIZE];
    __shared__ float sQ[2][SPEED_QR_SIZE * SPEED_QR_SIZE];
    __shared__ float sMul[SPEED_QR_SIZE * SPEED_QR_SIZE];
    __shared__ float sR[SPEED_QR_SIZE * SPEED_QR_SIZE];
    __shared__ float sVec[SPEED_QR_SIZE];

    for (int idx = tid; idx < size * size; idx += nthreads) {
        const int i = idx / size;
        const int j = idx % size;
        sA[idx] = A_in[idx];
        sTz[0][idx] = A_in[idx];
        sQ[0][idx] = (i == j) ? 1.0f : 0.0f;
    }
    __syncthreads();

    int cur = 0;
    for (int k = 0; k < size - 1; k++) {
        /* matrix_minor(tmp_z, k) */
        for (int idx = tid; idx < size * size; idx += nthreads) {
            const int i = idx / size;
            const int j = idx % size;
            if (i < k || j < k)
                sTz[cur][idx] = (i == j) ? 1.0f : 0.0f;
        }
        __syncthreads();

        /* matrix_take_kth_column(tmp_z, vec, k) */
        if (tid < size)
            sVec[tid] = sTz[cur][tid * size + k];
        __syncthreads();

        /* vector_norm -> vec[k] += sign*norm -> vector_div: sequential */
        if (tid == 0) {
            float norm = 0.0f;
            for (int i = 0; i < size; i++)
                norm = FADD(norm, FMUL(sVec[i], sVec[i]));
            norm = speed_sqrtd_roundf(norm);

            const int sgn = speed_get_sign(sA[k * size + k]);
            sVec[k] = FADD(sVec[k], FMUL((float)sgn, norm));

            float norm2 = 0.0f;
            for (int i = 0; i < size; i++)
                norm2 = FADD(norm2, FMUL(sVec[i], sVec[i]));
            norm2 = speed_sqrtd_roundf(norm2);

            for (int i = 0; i < size; i++)
                sVec[i] = FDIV(sVec[i], norm2);
        }
        __syncthreads();

        /* matrix_identity_minus_v_vt(tmp_q, vec) */
        for (int idx = tid; idx < size * size; idx += nthreads) {
            const int i = idx / size;
            const int j = idx % size;
            float val = FMUL(FMUL(-2.0f, sVec[i]), sVec[j]);
            if (i == j)
                val = FADD(val, 1.0f);
            sMul[idx] = val;
        }
        __syncthreads();

        /* tmp_mul = tmp_q * tmp_z; tmp_z = tmp_mul */
        for (int idx = tid; idx < size * size; idx += nthreads) {
            const int i = idx / size;
            const int j = idx % size;
            float acc = 0.0f;
            for (int m = 0; m < size; m++)
                acc = FADD(acc, FMUL(sMul[i * size + m], sTz[cur][m * size + j]));
            sTz[1 - cur][idx] = acc;
        }

        /* tmp_mul = tmp_q * Q; Q = tmp_mul */
        for (int idx = tid; idx < size * size; idx += nthreads) {
            const int i = idx / size;
            const int j = idx % size;
            float acc = 0.0f;
            for (int m = 0; m < size; m++)
                acc = FADD(acc, FMUL(sMul[i * size + m], sQ[cur][m * size + j]));
            sQ[1 - cur][idx] = acc;
        }
        __syncthreads();

        cur = 1 - cur;
        __syncthreads();
    }

    /* R = Q * A */
    for (int idx = tid; idx < size * size; idx += nthreads) {
        const int i = idx / size;
        const int j = idx % size;
        float acc = 0.0f;
        for (int m = 0; m < size; m++)
            acc = FADD(acc, FMUL(sQ[cur][i * size + m], sA[m * size + j]));
        sR[idx] = acc;
    }
    __syncthreads();

    for (int idx = tid; idx < size * size; idx += nthreads)
        R_out[idx] = sR[idx];

    /* tmp_rect = Q * B */
    for (int idx = tid; idx < total; idx += nthreads) {
        const int i = idx / num_blocks;
        const int j = idx % num_blocks;
        float acc = 0.0f;
        for (int m = 0; m < size; m++)
            acc = FADD(acc, FMUL(sQ[cur][i * size + m], B_in[m * num_blocks + j]));
        TmpRect_out[idx] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        /* is_matrix_regular(): `eigenvalues[i] < EIGENVALUE_EPS` in double */
        int bad = 0;
        for (int i = 0; i < size; i++) {
            if ((double)eigenvalues[i] < EIGENVALUE_EPS) {
                bad = 1;
                break;
            }
        }
        /* solve_triangular_system(): |R[i][i]| < EIGENVALUE_EPS */
        if (!bad) {
            for (int i = 0; i < size; i++) {
                if ((double)fabsf(sR[i * size + i]) < EIGENVALUE_EPS) {
                    bad = 1;
                    break;
                }
            }
        }
        *err_flag = bad;
    }
    __syncthreads();

    /* The CPU memsets linear_system_sol before the pointwise product (and the
     * triangular solve leaves the array untouched otherwise), so a zeroed
     * solution is the exact state est_params() continues with; the solve kernel
     * below is skipped through the flag. */
    if (*err_flag) {
        for (int idx = tid; idx < total; idx += nthreads)
            X_out[idx] = 0.0f;
    }
}

/* solve_triangular_system(): one thread per column j; the CPU solves the
 * columns in the same (independent) order. */
__global__ void speed_chroma_solve_kernel(const float *R, const float *TmpRect,
                                          float *X, int num_blocks,
                                          const int *err_flag)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_blocks) return;
    if (*err_flag) return; /* solution already zeroed by the QR kernel */

    for (int i = SPEED_QR_SIZE - 1; i >= 0; i--) {
        const float denominator = R[i * SPEED_QR_SIZE + i];
        float independent_term = TmpRect[i * num_blocks + j];
        for (int k = i + 1; k < SPEED_QR_SIZE; k++)
            independent_term = FSUB(independent_term,
                                    FMUL(X[k * num_blocks + j],
                                         R[i * SPEED_QR_SIZE + k]));
        X[i * num_blocks + j] = FDIV(independent_term, denominator);
    }
}

/* ------------------------------------------------------------------------- */
/* est_params, steps 5-6: pointwise product, sum of columns                   */
/* ------------------------------------------------------------------------- */

/* compute_pointwise_product_and_division(): X[i][j] = X[i][j]*Y[i][j]/B^2 */
__global__ void speed_chroma_pointwise_kernel(float *X, const float *Y,
                                              float denominator, int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    X[idx] = FDIV(FMUL(X[idx], Y[idx]), denominator);
}

/* sum_columns(): S[j] = sum_i X[i][j], accumulated in i order. The CPU stores
 * S in row 0 of the solution matrix; here it goes to a separate array. */
__global__ void speed_chroma_sum_columns_kernel(const float *X, float *S,
                                                int num_blocks,
                                                int elements_in_block)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_blocks) return;

    float acc = X[j];
    for (int i = 1; i < elements_in_block; i++)
        acc = FADD(acc, X[(size_t)i * num_blocks + j]);
    S[j] = acc;
}

} /* extern "C" */
