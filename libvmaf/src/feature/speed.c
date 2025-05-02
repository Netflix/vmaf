/**
 *
 *  Copyright 2016-2025 Netflix, Inc.
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

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "mem.h"
#include "opt.h"
#include "picture.h"
#include "picture_copy.h"
#include "vif_tools.h"

typedef struct SpeedDimensions {
    size_t original_height;
    size_t original_width;
    size_t scaled_height;
    size_t scaled_width;
    size_t alloc_height;
    size_t alloc_width;
    size_t operating_height;
    size_t operating_width;
    size_t block_size;
    size_t truncated_width;
    size_t truncated_height;
    size_t num_blocks_horizontal;
    size_t num_blocks_vertical;
    size_t num_blocks;
    size_t elements_in_block;
    size_t submatrix_width;
    size_t submatrix_height;
} SpeedDimensions;

typedef struct SpeedResultBuffers {
    float *entropies;
    float *variances;
} SpeedResultBuffers;

typedef struct SpeedBuffers {
    float *independent_term;
    float *linear_system_sol;
    float *cov_mat;
    float *eigenvalues;
    float *tmp_buffer;
} SpeedBuffers;

// Everything that is passed in as a feature option and is needed for
// SpEED computation
typedef struct SpeedOptions {
    double speed_kernelscale;
    double speed_prescale;
    char *speed_prescale_method;
    double speed_sigma_nn;
    double speed_nn_floor;
    int speed_weight_var_mode;
} SpeedOptions;

// Everything that is needed to compute SpEED given a pair of float buffers
// (ref, dis), except for what is provided in SpeedOptions
typedef struct SpeedState {
    SpeedDimensions dimensions;
    SpeedResultBuffers ref_results;
    SpeedResultBuffers dis_results;
    SpeedBuffers buffers;
    size_t float_stride;
} SpeedState;

#define DEFAULT_BLOCK_SIZE (5)
#define NUM_SQUARE_BUFFERS (5)
#define NUM_RECT_BUFFERS (1)
#define NUM_FRAME_BUFFERS (2)
#define NUM_SCALES (4)
#define EIGENVALUE_EPS (1e-6)
#define EIGENVALUE_MAX_ITERS (500)
#define ALMOST_EQUAL(x,c) (fabs((x) - (c)) < 1.0e-3)
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typedef struct Matrix {
    int rows;
    int cols;
    float *data;
} Matrix;

static void matrix_init(Matrix *mat, int rows, int cols, float *buffer)
{
    mat->data = buffer;
    mat->rows = rows;
    mat->cols = cols;
}

static void matrix_transpose(Matrix *m)
{
    // only works for square matrices due to being an in-place operation
    assert(m->rows == m->cols);
    int size = m->rows;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < i; j++) {
            float temp = m->data[i * size + j];
            m->data[i * size + j] = m->data[j * size + i];
            m->data[j * size + i] = temp;
        }
    }
}

static void matrix_zero(Matrix *m)
{
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] = 0;
        }
    }
}

static void matrix_identity(Matrix *m)
{
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] = (i == j ? 1 : 0);
        }
    }
}

static void matrix_copy(Matrix *dst, const Matrix *src)
{
    dst->rows = src->rows;
    dst->cols = src->cols;
    for (int i = 0; i < dst->rows; i++) {
        for (int j = 0; j < dst->cols; j++) {
            dst->data[i * dst->cols + j] = src->data[i * src->cols + j];
        }
    }
}

static void matrix_mul(Matrix *dst, const Matrix *x, const Matrix *y)
{
    assert(x->cols == y->rows);
    matrix_zero(dst);
    for (int i = 0; i < x->rows; i++) {
        for (int k = 0; k < x->cols; k++) {
            for (int j = 0; j < y->cols; j++) {
                dst->data[i * dst->cols + j] +=
                    x->data[i * x->cols + k] * y->data[k * y->cols + j];
            }
        }
    }
}

static void matrix_minor(Matrix *mat, int d)
{
    // only works for square matrices
    assert(mat->rows == mat->cols);
    int size = mat->rows;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i < d || j < d) {
                if (i == j)
                    mat->data[i * size + j] = 1;
                else
                    mat->data[i * size + j] = 0;
            }
        }
    }
}

static void matrix_identity_minus_v_vt(Matrix *dst, float *v)
{
    // dst = I - v v^T
    // only works for square matrices
    assert(dst->rows == dst->cols);
    int size = dst->rows;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dst->data[i * size + j] = -2 * v[i] * v[j];
        }
    }

    for (int i = 0; i < size; i++)
        dst->data[i * size + i] += 1;
}

static float vector_norm(const float *x, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

static void vector_div(const float *x, float d, float *y, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] / d;
}

static void matrix_take_kth_column(const Matrix *mat, float *v, int k)
{
    //take k-th column of m, put in v
    for (int i = 0; i < mat->rows; i++)
        v[i] = mat->data[i * mat->cols + k];
}

static int get_sign(float x)
{
    return (x >= 0 ? 1 : -1);
}

static float pythagoras(float x, float y)
{
    //hypotenuse length of the right triangle with sides x and y
    return sqrt(x * x + y * y);
}

static float compute_column_norm(float *A, int col, int start_row, int size)
{
    // euclidean norm of the column vector A[start_row:size, col]
    float norm = 0;
    for (int i = start_row; i < size; i++)
        norm += A[i * size + col] * A[i * size + col];
    return sqrt(norm);
}

// Compute a householder transformation (tau,v) of a vector
// x so that P x = [ I - tau*v*v' ] x annihilates x(1:n-1)
//
// On output, v is normalized so that v[0] = 1. The 1 is
// not actually stored; instead v[0] = -sign(x[0])*||x|| so
// that:
//              P x = v[0] * e_1

static float compute_householder_transform(float *A, int col, int start_row,
                                           int size)
{
    if (size - start_row == 1)
        return 0.0f;
    float xnorm = compute_column_norm(A, col, start_row + 1, size);
    if (xnorm == 0.0f)
        return 0.0f;
    float alpha = A[start_row * size + col];
    float beta = -get_sign(alpha) * pythagoras(alpha, xnorm);
    float tau = (beta - alpha) / beta;
    float s = alpha - beta;
    if (s != 0.0f) {
        for (int i = start_row; i < size; i++)
            A[i * size + col] /= s;
        A[start_row * size + col] = beta;
    }
    return tau;
}

// x = tau_i + A * v
// NOTE: only the rows and columns [start:size] of the involved matrices and
// vectors are used
static void tridiagonal_multiply(float *A, float *v, float *x, float tau_i,
                                 int start, int size)
{
    for (int i = start; i < size; i++) {
        x[i] = 0.0f;
        for (int j = start; j < size; j++)
            x[i] += tau_i * A[i * size + j] * v[j];
    }
}

// returns dot(x, v)
// NOTE: only the rows and columns [start:size] of the involved matrices and
// vectors are used
static float tridiagonal_dot_product(float *x, float *v, int start, int size)
{
    float res = 0;
    for (int i = start; i < size; i++)
        res += x[i] * v[i];
    return res;
}

// x += alpha * v
// NOTE: only the rows and columns [start:size] of the involved matrices and
// vectors are used
static void tridiagonal_axpy(float *x, float *v, float alpha, int start, int size)
{
    for (int i = start; i < size; i++)
        x[i] += alpha * v[i];
}

// A -= x * v' + v * x'
// NOTE: only the rows and columns [start:size] of the involved matrices and
// vectors are used
static void tridiagonal_syr2(float *A, float *x, float *v, int start, int size)
{
    for (int i = start; i < size; i++) {
        for (int j = start; j < size; j++) {
            A[i * size + j] -= x[i] * v[j] + v[i] * x[j];
        }
    }
}

// Adapted gsl_linalg_symmtd_decomp:
// https://github.com/ampl/gsl/blob/master/linalg/symmtd.c
static void convert_to_tridiagonal(float *A, int size, float *d, float *sd,
                                   float *buffer)
{
    float *v = buffer; buffer += size;
    float *x = buffer; buffer += size;

    // We apply N-2 Householder transformations to zero out the elements
    // outside of the diagonal or subdiagonal of each of the N-2 first columns
    for (int i = 0; i < size - 2; i++) {
        // Compute the vector v of the Householder transform that
        // annihilates A[i+2:size, i]
        float tau_i = compute_householder_transform(A, i, i + 1, size);

        // Copy i'th subcolumn of a into v
        for (int j = i + 1; j < size; j++)
            v[j] = A[j * size + i];

        if (tau_i != 0.0f) {
            // Set the first element of the returned vector to 1,
            // since compute_householder_transform does this implicitly
            A[(i + 1) * size + i] = v[i + 1];
            v[i + 1] = 1.0f;
            // All operations described here are applied only to the rows and
            // columns in [i+1:size]
            // x = tau_i * A * v
            tridiagonal_multiply(A, v, x, tau_i, i + 1, size);
            // x -= 0.5 * tau_i * dot(x, v) * v]
            float xv = tridiagonal_dot_product(x, v, i + 1, size);
            float alpha = -0.5 * tau_i * xv;
            tridiagonal_axpy(x, v, alpha, i + 1, size);
            // A = A - v * x' - x * v'
            tridiagonal_syr2(A, x, v, i + 1, size);
        }
    }

    // Copy the diagonal and subdiagonal elements into d and sd, respectively
    for (int i = 0; i < size; i++)
        d[i] = A[i * size + i];
    for (int i = 0; i < size - 1; i++)
        sd[i] = A[(i + 1) * size + i];
}

static void chop_small_elements(float *d, float *sd, int size)
{
    for (int i = 0; i < size - 1; i++) {
        if (fabsf(sd[i]) < EIGENVALUE_EPS * (fabsf(d[i]) + fabsf(d[i + 1])))
            sd[i] = 0.0f;
    }
}

static float trailing_eigenvalue(float *d, float *sd, int n)
{
    float ta = d[n - 2];
    float tb = d[n - 1];
    float tab = sd[n - 2];
    float dt = (ta - tb) / 2.0;

    if (dt > 0)
        return tb - tab * (tab / (dt + pythagoras(dt, tab)));
    else if (dt == 0)
        return tb - fabsf(tab);
    else
        return tb + tab * (tab / (-dt + pythagoras(dt, tab)));
}

static void create_givens(const float a, const float b, float *c, float *s)
{
    if (b == 0) {
        *c = 1;
        *s = 0;
    } else if (fabsf(b) > fabsf(a)) {
        float t = -a / b;
        float s1 = 1.0 / sqrt(1 + t * t);
        *s = s1;
        *c = s1 * t;
    } else {
        float t = -b / a;
        float c1 = 1.0 / sqrt(1 + t * t);
        *c = c1;
        *s = c1 * t;
    }
}

static void qr_step(float *d, float *sd, int n)
{
    float mu = trailing_eigenvalue(d, sd, n);
    if (EIGENVALUE_EPS * fabsf(mu) > fabsf(d[0]) + fabsf(sd[0]))
        mu = 0;

    float x = d[0] - mu;
    float z = sd[0];

    float ak = 0;
    float bk = 0;
    float zk = 0;

    float ap = d[0];
    float bp = sd[0];

    float aq = d[1];

    if (n == 2) {
        float c, s;
        create_givens(x, z, &c, &s);

        float ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        float bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        float aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);

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
        create_givens(x, z, &c, &s);

        /* compute G' T G */
        float bk1 = c * bk - s * zk;

        float ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        float bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        float zp1 = -s * bq;

        float aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);
        float bq1 = c * bq;

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

        if (k < n - 2) {
            sd[k + 1] = bp;
        }

        x = bk;
        z = zk;
    }

    d[n - 1] = ap;
    sd[n - 2] = bk;
}

// Adapted gsl_eigen_symm: https://github.com/ampl/gsl/blob/master/eigen/symm.c
static void compute_eigenvalues_tridiagonal(float *d, float *sd,
                                            float *eigenvalues, int size)
{
    // Initial pass to remove subdiagonal elements which are effectively zero
    chop_small_elements(d, sd, size);

    // Progressively reduce the matrix until it is diagonal
    int b = size - 1;
    int iter = 0;
    while (b > 0 && iter < EIGENVALUE_MAX_ITERS) {
        if (sd[b - 1] == 0.0f) {
            b--;
            continue;
        }
        // Find the largest unreduced block (a, b) starting from b and working
        // backwards
        int a = b - 1;
        while (a > 0) {
            if (sd[a - 1] == 0.0f) {
                break;
            }
            a--;
        }

        const int n_block = b - a + 1;
        float *d_block = d + a;
        float *sd_block = sd + a;
        qr_step(d_block, sd_block, n_block);
        chop_small_elements(d_block, sd_block, n_block);
        iter++;
    }

    if (iter == EIGENVALUE_MAX_ITERS) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "compute_eigenvalues_tridiagonal: max iterations reached, "
                 "possible non-convergence\n");
    }

    for (int i = 0; i < size; i++)
        eigenvalues[i] = d[i];
}

static void compute_eigenvalues(float *A_immutable, float *eigenvalues,
                                int size, float *buffer)
{
    float *A = buffer; buffer += size * size;
    float *d = buffer; buffer += size;
    float *sd = buffer; buffer += size;
    float *tmp = buffer; buffer += 2 * size;
    // Operate on a copy of the matrix
    memcpy(A, A_immutable, size * size * sizeof(float));
    // Handle special case
    if (size == 1) {
        eigenvalues[0] = A[0 * size + 0];
        return;
    }
    convert_to_tridiagonal(A, size, d, sd, tmp);
    compute_eigenvalues_tridiagonal(d, sd, eigenvalues, size);
}

// Implementation of the QR decomposition algorithm with Householder
// reflections for an arbitrary square matrix
// https://www.cs.utexas.edu/users/flame/Notes/NotesOnHouseholderQR.pdf
static void matrix_qr_decomposition(Matrix *A, Matrix *Q, Matrix *R,
                                    Matrix *tmp_q, Matrix *tmp_z)
{
    assert(A->rows == A->cols);
    int size = A->rows;

    // We need 3 temporary matrices. We can use R as a temporary matrix during
    // the process since it's only used at the end. We can also use R for the
    // temporary vector, since both uses are disjoint in time
    Matrix *tmp_mul = R;
    float *vec = R->data;

    // Initially, tmp_z = A
    matrix_copy(tmp_z, A);

    // Q starts as the identity, and is left-multiplied by tmp_q at each iteration
    matrix_identity(Q);

    for (int k = 0; k < size - 1; k++) {
        matrix_minor(tmp_z, k);
        matrix_take_kth_column(tmp_z, vec, k);

        float norm = vector_norm(vec, size);
        int sign = get_sign(A->data[k * size + k]);
        vec[k] += sign * norm;

        vector_div(vec, vector_norm(vec, size), vec, size);
        matrix_identity_minus_v_vt(tmp_q, vec);

        matrix_mul(tmp_mul, tmp_q, tmp_z);
        matrix_copy(tmp_z, tmp_mul);
        matrix_mul(tmp_mul, tmp_q, Q);
        matrix_copy(Q, tmp_mul);
    }

    matrix_mul(R, Q, A);
    matrix_transpose(Q);
}

// Solves RX = B, where R is square, upper triangular and invertible
// Uses backward substitution algorithm
static int solve_triangular_system(const Matrix *R, Matrix *X, const Matrix *B)
{
    assert(R->rows == R->cols);
    assert(R->rows == X->rows);
    assert(R->rows == B->rows);
    assert(X->cols == B->cols);

    for (int i = R->rows - 1; i >= 0; i--) {
        float denominator = R->data[i * R->cols + i];
        if (fabsf(denominator) < EIGENVALUE_EPS) {
            return -EINVAL;
        }
        for (int j = 0; j < X->cols; j++) {
            float independent_term = B->data[i * B->cols + j];
            for (int k = i + 1; k < R->rows; k++) {
                independent_term -=
                    (X->data[k * X->cols + j] * R->data[i * R->cols + k]);
            }
            X->data[i * X->cols + j] = independent_term / denominator;
        }
    }
    return 0;
}

// Solves the linear system A X = B, using the QR decomposition of A
static int solve_linear_system(float *A_data, int A_size, float *B_data,
                               int B_cols, float *output_data,
                               float *tmp_buffer)
{
    float *A_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *Q_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *R_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *tmp1_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *tmp2_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *tmp_rect_buffer = tmp_buffer; tmp_buffer += A_size * B_cols;

    Matrix A_immutable; matrix_init(&A_immutable, A_size, A_size, A_data);
    Matrix A; matrix_init(&A, A_size, A_size, A_buffer);
    matrix_copy(&A, &A_immutable);

    Matrix Q; matrix_init(&Q, A_size, A_size, Q_buffer);
    Matrix R; matrix_init(&R, A_size, A_size, R_buffer);
    Matrix tmp1; matrix_init(&tmp1, A_size, A_size, tmp1_buffer);
    Matrix tmp2; matrix_init(&tmp2, A_size, A_size, tmp2_buffer);

    Matrix B_immutable; matrix_init(&B_immutable, A_size, B_cols, B_data);

    Matrix tmp_rect; matrix_init(&tmp_rect, A_size, B_cols, tmp_rect_buffer);
    Matrix X; matrix_init(&X, A_size, B_cols, output_data);

    // A = Q R, where Q^{-1} = Q^T and R is upper triangular
    // A X = B  ==>  Q R X = B  ==>  R X = Q^T B
    matrix_qr_decomposition(&A, &Q, &R, &tmp1, &tmp2);
    matrix_transpose(&Q);
    matrix_mul(&tmp_rect, &Q, &B_immutable);
    int err = solve_triangular_system(&R, &X, &tmp_rect);
    return err;
}

static float compute_mean(SpeedDimensions dim, const float *data,
                          size_t stride_px, int start_row, int start_col)
{
    float result = 0;
    for (size_t i = 0; i < dim.submatrix_height; i++) {
        for (size_t j = 0; j < dim.submatrix_width; j++)
            result += data[(start_row + i) * stride_px + (start_col + j)];
    }

    return result / (dim.submatrix_width * dim.submatrix_height);
}

// Computes the covariance between two arrays of the same given size
// The arrays are stored as submatrices of the input data, starting at
// (start_row_x, start_col_x) and (start_row_y, start_col_y)
// and having dimensions (submatrix_height, submatrix_width)
static float compute_covariance(SpeedDimensions dim, const float *data,
                                const float *means, size_t stride_px,
                                int start_row_x, int start_col_x,
                                int start_row_y, int start_col_y)
{
    double mean_x = means[start_row_x * dim.block_size + start_col_x];
    double mean_y = means[start_row_y * dim.block_size + start_col_y];
    double result = 0;
    for (size_t i = 0; i < dim.submatrix_height; i++) {
        for (size_t j = 0; j < dim.submatrix_width; j++) {
            double val_x =
                data[(start_row_x + i) * stride_px + (start_col_x + j)];
            double val_y =
                data[(start_row_y + i) * stride_px + (start_col_y + j)];
            result += (val_x - mean_x) * (val_y - mean_y);
        }
    }
    return result / (dim.submatrix_width * dim.submatrix_height);
}

static void compute_covariance_matrix(SpeedDimensions dim, const float *data,
                                      float *cov_mat, float *means,
                                      size_t stride_px)
{
    for (size_t start_row = 0; start_row < dim.block_size; start_row++) {
        for (size_t start_col = 0; start_col < dim.block_size; start_col++) {
            means[start_row * dim.block_size + start_col] =
                compute_mean(dim, data, stride_px, start_row, start_col);
        }
    }
    size_t elements_in_block = dim.block_size * dim.block_size;

    for (size_t x_index = 0; x_index < dim.elements_in_block; x_index++) {
        for (size_t y_index = 0; y_index <= x_index; y_index++) {
            size_t start_row_x = x_index / dim.block_size;
            size_t start_col_x = x_index % dim.block_size;
            size_t start_row_y = y_index / dim.block_size;
            size_t start_col_y = y_index % dim.block_size;
            float covariance =
                compute_covariance(dim, data, means, stride_px, start_row_x,
                                   start_col_x, start_row_y, start_col_y);
            cov_mat[x_index * elements_in_block + y_index] = covariance;
            cov_mat[y_index * elements_in_block + x_index] = covariance;
        }
    }
}

static void compute_independent_term(SpeedDimensions dim, const float *data,
                                     float *independent_term, size_t stride_px)
{
    for (size_t start_i = 0; start_i < dim.block_size; start_i++) {
        for (size_t start_j = 0; start_j < dim.block_size; start_j++) {
            for (size_t i = start_i; i < dim.truncated_height; i += dim.block_size) {
                for (size_t j = start_j; j < dim.truncated_width; j += dim.block_size) {
                    size_t out_row = start_i * dim.block_size + start_j;
                    size_t out_col =
                        (((i - start_i) / dim.block_size) *
                            dim.num_blocks_horizontal) + ((j - start_j) / dim.block_size);
                    independent_term[out_row * dim.num_blocks + out_col] =
                        data[i * stride_px + j];
                }
            }
        }
    }
}

static void compute_pointwise_product_and_division(SpeedDimensions dim,
                                                   float *X, const float *Y,
                                                   float denominator)
{
    for (size_t i = 0; i < dim.elements_in_block; i++) {
        for (size_t j = 0; j < dim.num_blocks; j++) {
            X[i * dim.num_blocks + j] =
                (X[i * dim.num_blocks + j] * Y[i * dim.num_blocks + j]) / denominator;
        }
    }
}

static void sum_columns(SpeedDimensions dim, float *X)
{
    for (size_t i = 1; i < dim.elements_in_block; i++) {
        for (size_t j = 0; j < dim.num_blocks; j++) {
            X[0 * dim.num_blocks + j] += X[i * dim.num_blocks + j];
        }
    }
}

static void update_entropy(SpeedDimensions dim, float *entropy, const float *S,
                           float L, float sigma_nn)
{
    for (size_t i = 0; i < dim.num_blocks_vertical; i++) {
        for (size_t j = 0; j < dim.num_blocks_horizontal; j++) {
            entropy[i * dim.num_blocks_horizontal + j] +=
                log2(L * S[i * dim.num_blocks_horizontal + j] + sigma_nn) +
                log2(2 * M_PI * M_E);
        }
    }
}

static bool is_matrix_regular(SpeedDimensions dim, const float *eigenvalues)
{
    for (size_t i = 0; i < dim.elements_in_block; i++) {
        if (eigenvalues[i] < EIGENVALUE_EPS) {
            return false;
        }
    }
    return true;
}

static int est_params(SpeedState *s, const float *data, float sigma_nn,
                      SpeedResultBuffers *output)
{
    SpeedDimensions dim = s->dimensions;
    size_t stride_px = s->float_stride / sizeof(float);

    // Step 1: Compute the covariance matrix K
    // We use the eigenvalues array as a temporary array to store the means
    // needed for the covariance
    compute_covariance_matrix(dim, data, s->buffers.cov_mat,
                              s->buffers.eigenvalues, stride_px);

    // Step 2: Compute the eigenvalues of the covariance matrix
    compute_eigenvalues(s->buffers.cov_mat, s->buffers.eigenvalues,
                        dim.elements_in_block, s->buffers.tmp_buffer);

    // Step 3: Compute independent term for the linear system
    compute_independent_term(dim, data, s->buffers.independent_term, stride_px);

    // Step 4: Solve the linear equation KX = Y, where K is the covariance
    // matrix and Y is the independent term matrix above
    int err = 0;
    bool regular = is_matrix_regular(dim, s->buffers.eigenvalues);
    if (regular) {
        err = solve_linear_system(s->buffers.cov_mat, dim.elements_in_block,
                                  s->buffers.independent_term,
                                  dim.num_blocks, s->buffers.linear_system_sol,
                                  s->buffers.tmp_buffer);
    }

    bool cannot_invert = !regular || err;
    if (cannot_invert) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "est_params: covariance matrix is singular\n");
        memset(s->buffers.linear_system_sol, 0,
               dim.elements_in_block * dim.num_blocks * sizeof(float));
    }

    // Step 5: Compute the pointwise product Z = (X * Y)/B^2, where X and Y are
    // from the linear system above, and B is the block size.
    // Store the results in s->linear_system_sol
    compute_pointwise_product_and_division(dim, s->buffers.linear_system_sol,
                                           s->buffers.independent_term,
                                           dim.elements_in_block);

    // Step 6: Sum each column in Z into an array S.
    // Store the results in the first row of s->linear_system_sol
    sum_columns(dim, s->buffers.linear_system_sol);

    // Step 7: Reshape S into a matrix of dimensions (H/B, W/B)
    // This is a no-op, we just need to index it with the right dimensions

    // Step 8: Construct a zeroed-out matrix E of size (H/B, W/B)
    memset(output->entropies, 0,
           dim.num_blocks_horizontal * dim.num_blocks_vertical * sizeof(float));

    // Step 9: For each eigenvalue L, update the entropy matrix
    for (size_t k = 0; k < dim.elements_in_block; k++) {
        float L = s->buffers.eigenvalues[k] < 0 ? 0 : s->buffers.eigenvalues[k];
        update_entropy(dim, output->entropies, s->buffers.linear_system_sol, L,
                       sigma_nn);
    }

    // Step 10: Return S, E
    memcpy(output->variances, s->buffers.linear_system_sol,
           dim.num_blocks * sizeof(float));

    return cannot_invert ? -EINVAL : 0;
}

static float get_speed_score(SpeedDimensions dim, SpeedResultBuffers ref_results,
                             SpeedResultBuffers dis_results, float sigma_nn,
                             float nn_floor, int speed_weight_var_mode)
{
    float score = 0;
    float base_entropy = dim.elements_in_block * (log2((1 + nn_floor) * sigma_nn) + log2(2 * M_PI * M_E));
    for (size_t i = 0; i < dim.num_blocks; i++) {
        if ((ref_results.entropies[i] < base_entropy) &&
            (dis_results.entropies[i] < base_entropy))
        {
            // If both entropies are below the base_entropy,
            // there is no visible difference
            score += 0;
        } else {
            float spatial_ref = 0.0;
            float spatial_dis = 0.0;
            if (speed_weight_var_mode == 0) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + dis_results.variances[i]);
            } else if (speed_weight_var_mode == 1) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + ref_results.variances[i]);
            } else if (speed_weight_var_mode == 2) {
                spatial_ref = ref_results.entropies[i] * log2(1 + dis_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + dis_results.variances[i]);
            } else if (speed_weight_var_mode == 3) {
                spatial_ref = ref_results.entropies[i] * log2(1 + (ref_results.variances[i] + dis_results.variances[i]) / 2.0);
                spatial_dis = dis_results.entropies[i] * log2(1 + (ref_results.variances[i] + dis_results.variances[i]) / 2.0);
            } else if (speed_weight_var_mode == 4) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + (ref_results.variances[i] + dis_results.variances[i]) / 2.0);
            } else if (speed_weight_var_mode == 5) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + (0.75 * ref_results.variances[i] + 0.25 * dis_results.variances[i]));
            } else if (speed_weight_var_mode == 6) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + (0.25 * ref_results.variances[i] + 0.75 * dis_results.variances[i]));
            } else {
                return -EINVAL;
            }
            score += fabsf(spatial_ref - spatial_dis);
        }

    }

    return score / dim.num_blocks;
}

static void subtract_image(float *im1, float *im2, int w, int h, size_t stride) {
    size_t stride_px = stride / sizeof(float);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            im1[i * stride_px + j] -= im2[i * stride_px + j];
        }
    }
}

// Filters the image with a Gaussian filter and then performs local
// mean subtraction
static void filter_and_downscale(SpeedDimensions dim, SpeedOptions *opt,
                                 float *frame_buffer, float *tmp_buffer,
                                 size_t float_stride)
{
    size_t stride_px = float_stride / sizeof(float);

    size_t frame_size = stride_px * dim.alloc_height;
    float *curr_scale = tmp_buffer; tmp_buffer += frame_size;
    float *tmpbuf = tmp_buffer; tmp_buffer += frame_size;

    // The scaling method has been checked for validity in the init callback
    enum vif_scaling_method scaling_method;
    vif_get_scaling_method(opt->speed_prescale_method, &scaling_method);

    if (!ALMOST_EQUAL(opt->speed_prescale, 1.0)) {
        memcpy(tmpbuf, frame_buffer,
               stride_px * dim.alloc_height * sizeof(float));
        vif_scale_frame_s(scaling_method, tmpbuf, frame_buffer,
                          dim.original_width, dim.original_height, stride_px,
                          dim.scaled_width, dim.scaled_height, stride_px);
    }

    // The kernelscale has been checked for validity in the init callback
    int filter_width_antialias = vif_get_filter_size(1, opt->speed_kernelscale);
    float filter_antialias[128];
    speed_get_antialias_filter(filter_antialias, NUM_SCALES,
                               opt->speed_kernelscale);
    vif_filter1d_s(filter_antialias, frame_buffer, curr_scale, tmpbuf,
                   dim.scaled_width, dim.scaled_height, float_stride,
                   float_stride, filter_width_antialias);

    vif_dec16_s(curr_scale, frame_buffer, dim.scaled_width, dim.scaled_height,
                float_stride, float_stride);

    size_t downscaled_w = dim.scaled_width >> NUM_SCALES;
    size_t downscaled_h = dim.scaled_height >> NUM_SCALES;

    int filter_width = vif_get_filter_size(NUM_SCALES, opt->speed_kernelscale);
    float filter[128];
    vif_get_filter(filter, NUM_SCALES, opt->speed_kernelscale);
    vif_filter1d_s(filter, frame_buffer, curr_scale, tmpbuf, downscaled_w,
                   downscaled_h, float_stride, float_stride, filter_width);
    subtract_image(frame_buffer, curr_scale, downscaled_w, downscaled_h,
                   float_stride);
}

int speed_extract_score(SpeedState *s, SpeedOptions *opt, float *ref,
                        float *dis, float *score)
{
    filter_and_downscale(s->dimensions, opt, ref, s->buffers.tmp_buffer,
                         s->float_stride);
    int err_ref = est_params(s, ref, opt->speed_sigma_nn, &(s->ref_results));

    filter_and_downscale(s->dimensions, opt, dis, s->buffers.tmp_buffer,
                         s->float_stride);

    int err_dis = est_params(s, dis, opt->speed_sigma_nn, &(s->dis_results));

    // If only one of ref and dis was numerically unstable (very rare)
    // we return 0 instead of an inflated score that may skew the average
    if ((err_ref && !err_dis) || (!err_ref && err_dis)) {
        *score = 0.0f;
    } else {
        *score = get_speed_score(s->dimensions, s->ref_results, s->dis_results,
                                 opt->speed_sigma_nn, opt->speed_nn_floor,
                                 opt->speed_weight_var_mode);
    }

    return err_ref || err_dis;
}

static int speed_init_dimensions(SpeedDimensions *dim, int w, int h,
                                 double speed_prescale)
{
    dim->original_height = h;
    dim->original_width = w;
    dim->scaled_height = (int)(dim->original_height * speed_prescale + 0.5);
    dim->scaled_width = (int)(dim->original_width * speed_prescale + 0.5);
    dim->alloc_height = MAX(dim->original_height, dim->scaled_height);
    dim->alloc_width = MAX(dim->original_width, dim->scaled_width);
    dim->operating_height = dim->scaled_height >> NUM_SCALES;
    dim->operating_width = dim->scaled_width >> NUM_SCALES;
    dim->block_size = DEFAULT_BLOCK_SIZE;
    dim->truncated_width =
        (dim->operating_width / dim->block_size) * dim->block_size;
    dim->truncated_height =
        (dim->operating_height / dim->block_size * dim->block_size);
    dim->num_blocks_horizontal = dim->truncated_width / dim->block_size;
    dim->num_blocks_vertical = dim->truncated_height / dim->block_size;
    dim->num_blocks = dim->num_blocks_horizontal * dim->num_blocks_vertical;
    dim->elements_in_block = dim->block_size * dim->block_size;
    dim->submatrix_width = dim->truncated_width - dim->block_size + 1;
    dim->submatrix_height = dim->truncated_height - dim->block_size + 1;

    if (dim->truncated_height == 0 || dim->truncated_width == 0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SpEED: image too small, operating width or height is 0");
        return -EINVAL;
    }
    return 0;
}

int speed_init(SpeedState *s, SpeedOptions *opt, int w, int h)
{
    SpeedDimensions *dim = &s->dimensions;
    speed_init_dimensions(dim, w, h, opt->speed_prescale);

    // Check that the kernelscale is valid
    if (!vif_validate_kernelscale(opt->speed_kernelscale)) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "invalid speed_kernelscale");
        return -EINVAL;
    }

    enum vif_scaling_method scaling_method;
    if (vif_get_scaling_method(opt->speed_prescale_method, &scaling_method)) {
        return -EINVAL;
    }

    s->float_stride = ALIGN_CEIL(dim->alloc_width * sizeof(float));
    size_t stride_px = s->float_stride / sizeof(float);

    size_t tmp_buffer_size = sizeof(float) * (
        NUM_SQUARE_BUFFERS * dim->elements_in_block * dim->elements_in_block
        + NUM_RECT_BUFFERS * dim->elements_in_block * dim->num_blocks
        + NUM_FRAME_BUFFERS * dim->alloc_height * stride_px
    );

    s->buffers.independent_term =
        aligned_malloc(sizeof(float) * dim->num_blocks * dim->elements_in_block, 32);
    if (!s->buffers.independent_term)
        return -ENOMEM;
    s->buffers.linear_system_sol =
        aligned_malloc(sizeof(float) * dim->num_blocks * dim->elements_in_block, 32);
    if (!s->buffers.linear_system_sol)
        return -ENOMEM;
    s->buffers.cov_mat =
        aligned_malloc(sizeof(float) * dim->elements_in_block * dim->elements_in_block, 32);
    if (!s->buffers.cov_mat)
        return -ENOMEM;
    s->buffers.eigenvalues =
        aligned_malloc(sizeof(float) * dim->elements_in_block, 32);
    if (!s->buffers.eigenvalues)
        return -ENOMEM;
    s->buffers.tmp_buffer = aligned_malloc(tmp_buffer_size, 32);
    if (!s->buffers.tmp_buffer)
        return -ENOMEM;

    s->ref_results.entropies = aligned_malloc(sizeof(float) * dim->num_blocks, 32);
    if (!s->ref_results.entropies)
        return -ENOMEM;
    s->ref_results.variances = aligned_malloc(sizeof(float) * dim->num_blocks, 32);
    if (!s->ref_results.variances)
        return -ENOMEM;
    s->dis_results.entropies = aligned_malloc(sizeof(float) * dim->num_blocks, 32);
    if (!s->dis_results.entropies)
        return -ENOMEM;
    s->dis_results.variances = aligned_malloc(sizeof(float) * dim->num_blocks, 32);
    if (!s->dis_results.variances)
        return -ENOMEM;

    return 0;
}

int speed_close(SpeedState *s) {
    if (s->buffers.independent_term)
        aligned_free(s->buffers.independent_term);
    if (s->buffers.linear_system_sol)
        aligned_free(s->buffers.linear_system_sol);
    if (s->buffers.cov_mat)
        aligned_free(s->buffers.cov_mat);
    if (s->buffers.eigenvalues)
        aligned_free(s->buffers.eigenvalues);
    if (s->buffers.tmp_buffer)
        aligned_free(s->buffers.tmp_buffer);

    if (s->ref_results.entropies)
        aligned_free(s->ref_results.entropies);
    if (s->ref_results.variances)
        aligned_free(s->ref_results.variances);
    if (s->dis_results.entropies)
        aligned_free(s->dis_results.entropies);
    if (s->dis_results.variances)
        aligned_free(s->dis_results.variances);

    return 0;
}

#define DEFAULT_SPEED_SIGMA_NN (0.29)
#define DEFAULT_SPEED_MAX_VAL (1000.0)
#define DEFAULT_SPEED_NN_FLOOR (0.0)
#define DEFAULT_SPEED_KERNELSCALE (1.0)
#define DEFAULT_SPEED_PRESCALE (1.0)
#define DEFAULT_SPEED_PRESCALE_METHOD ("nearest")
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct SpeedChromaState {
    SpeedState speed_state;
    SpeedOptions speed_options;
    float *frame_buffer_ref;
    float *frame_buffer_dis;
    VmafDictionary *feature_name_dict;
    double speed_chroma_kernelscale;
    double speed_chroma_prescale;
    char *speed_chroma_prescale_method;
    double speed_chroma_sigma_nn;
    double speed_chroma_nn_floor;
    double speed_chroma_max_val;
    int speed_weight_var_mode;
} SpeedChromaState;

static const VmafOption options_chroma[] = {
    {
        .name = "speed_kernelscale",
        .help = "scaling factor for the gaussian kernel (2.0 means "
                "multiplying the standard deviation by 2 and enlarge "
                "the kernel size accordingly",
        .offset = offsetof(SpeedChromaState, speed_chroma_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_KERNELSCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ks",
    },
    {
        .name = "speed_prescale",
        .help = "scaling factor for the frame (2.0 means "
                "making the image twice as large on each dimension)",
        .offset = offsetof(SpeedChromaState, speed_chroma_prescale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_PRESCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ps",
    },
    {
        .name = "speed_prescale_method",
        .help = "scaling method for the frame, supported options: "
                 "[nearest, bilinear, bicubic, lanczos4]",
        .offset = offsetof(SpeedChromaState, speed_chroma_prescale_method),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_SPEED_PRESCALE_METHOD,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "psm",
    },
    {
        .name = "speed_sigma_nn",
        .help = "standard deviation of neural noise",
        .offset = offsetof(SpeedChromaState, speed_chroma_sigma_nn),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_SIGMA_NN,
        .min = 0.1,
        .max = 2.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "snn",
    },
    {
        .name = "speed_nn_floor",
        .help = "neural noise floor, expressed in percentage of sigma_nn",
        .offset = offsetof(SpeedChromaState, speed_chroma_nn_floor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_NN_FLOOR,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "nnf",
    },
    {
        .name = "speed_max_val",
        .help = "maximum value allowed; "
                "larger values will be clipped to this value",
        .offset = offsetof(SpeedChromaState, speed_chroma_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mxv",
    },
    {
        .name = "speed_weight_var_mode",
        .help = "different approaches to perform variance-absed weighting",
        .offset = offsetof(SpeedChromaState, speed_weight_var_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.d = 0,
        .min = 0,
        .max = 6,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "wvm",
    },
    { 0 }
};

static int init_chroma(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                       unsigned bpc, unsigned w, unsigned h)
{
    (void)bpc;

    switch (pix_fmt) {
        case VMAF_PIX_FMT_UNKNOWN:
        case VMAF_PIX_FMT_YUV400P:
            return -EINVAL;
        case VMAF_PIX_FMT_YUV420P:
            w /= 2;
            h /= 2;
            break;
        case VMAF_PIX_FMT_YUV422P:
            w /= 2;
            break;
        case VMAF_PIX_FMT_YUV444P:
            break;
    }

    SpeedChromaState *s = fex->priv;
    s->speed_options = (SpeedOptions) {
        .speed_kernelscale = s->speed_chroma_kernelscale,
        .speed_prescale = s->speed_chroma_prescale,
        .speed_prescale_method = s->speed_chroma_prescale_method,
        .speed_sigma_nn = s->speed_chroma_sigma_nn,
        .speed_nn_floor = s->speed_chroma_nn_floor,
        .speed_weight_var_mode = s->speed_weight_var_mode,
    };
    speed_init(&s->speed_state, &s->speed_options, w, h);
    SpeedDimensions dim = s->speed_state.dimensions;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    s->frame_buffer_ref =
        aligned_malloc(s->speed_state.float_stride * dim.alloc_height, 32);
    if (!s->frame_buffer_ref)
        return -ENOMEM;
    s->frame_buffer_dis =
        aligned_malloc(s->speed_state.float_stride * dim.alloc_height, 32);
    if (!s->frame_buffer_dis)
        return -ENOMEM;

    return 0;
}

static float extract_channel(SpeedChromaState *s, VmafPicture *ref_pic,
                             VmafPicture *dist_pic, int channel, float *score)
{
    picture_copy(s->frame_buffer_ref, s->speed_state.float_stride,
                 ref_pic, -128, ref_pic->bpc, channel);
    picture_copy(s->frame_buffer_dis, s->speed_state.float_stride,
                 dist_pic, -128, dist_pic->bpc, channel);
    return speed_extract_score(&s->speed_state, &s->speed_options,
                               s->frame_buffer_ref, s->frame_buffer_dis, score);
}

static int extract_chroma(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    (void)ref_pic_90;
    (void)dist_pic_90;

    SpeedChromaState *s = fex->priv;

    float score_u, score_v;
    int err_u = extract_channel(s, ref_pic, dist_pic, 1, &score_u);
    int err_v = extract_channel(s, ref_pic, dist_pic, 2, &score_v);

    // There are edge cases where one or both channels (U and V) have singular
    // covariance matrices. For example, when the channel is completely flat.
    // If only one channel is singular, we impute its score from the other
    // channel, and therefore the combined score_uv is equal to the other
    // channel. This is a better approximation than imputing it to be zero.

    float score_uv;
    if (err_u && !err_v) {
        score_uv = score_v;
    } else if (err_v && !err_u) {
        score_uv = score_u; }
    else {
        score_uv = (score_u + score_v) / 2.0;
    }

    int err = 0;

    err |=
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "Speed_chroma_feature_speed_chroma_u_score",
             MIN(score_u, s->speed_chroma_max_val), index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "Speed_chroma_feature_speed_chroma_v_score",
            MIN(score_v, s->speed_chroma_max_val), index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "Speed_chroma_feature_speed_chroma_uv_score",
            MIN(score_uv, s->speed_chroma_max_val), index);
    return err;
}

static int close_chroma(VmafFeatureExtractor *fex)
{
    SpeedChromaState *s = fex->priv;

    speed_close(&s->speed_state);

    if (s->frame_buffer_ref)
        aligned_free(s->frame_buffer_ref);
    if (s->frame_buffer_dis)
        aligned_free(s->frame_buffer_dis);

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    return 0;
}

static const char *provided_features_chroma[] = {
    "Speed_chroma_feature_speed_chroma_u_score",
    "Speed_chroma_feature_speed_chroma_v_score",
    "Speed_chroma_feature_speed_chroma_uv_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_speed_chroma = {
    .name = "speed_chroma",
    .init = init_chroma,
    .extract = extract_chroma,
    .close = close_chroma,
    .options = options_chroma,
    .priv_size = sizeof(SpeedChromaState),
    .provided_features = provided_features_chroma,
};

#define DEFAULT_SPEED_SIGMA_NN (0.29)
#define DEFAULT_SPEED_MAX_VAL (1000.0)
#define DEFAULT_SPEED_NN_FLOOR (0.0)
#define DEFAULT_SPEED_KERNELSCALE (1.0)
#define DEFAULT_SPEED_PRESCALE (1.0)
#define DEFAULT_SPEED_PRESCALE_METHOD ("nearest")

typedef struct SpeedTemporalState {
    SpeedState speed_state;
    SpeedOptions speed_options;
    float *frame_buffer_ref[2];
    float *frame_buffer_dis[2];
    VmafDictionary *feature_name_dict;
    int index;
    double score;
    double speed_temporal_kernelscale;
    double speed_temporal_prescale;
    char *speed_temporal_prescale_method;
    double speed_temporal_sigma_nn;
    double speed_temporal_nn_floor;
    double speed_temporal_max_val;
    bool speed_temporal_use_ref_diff;
} SpeedTemporalState;

static const VmafOption options[] = {
    {
        .name = "speed_kernelscale",
        .help = "scaling factor for the gaussian kernel (2.0 means "
                "multiplying the standard deviation by 2 and enlarge "
                "the kernel size accordingly",
        .offset = offsetof(SpeedTemporalState, speed_temporal_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_KERNELSCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ks",
    },
    {
        .name = "speed_prescale",
        .help = "scaling factor for the frame (2.0 means "
                "making the image twice as large on each dimension)",
        .offset = offsetof(SpeedTemporalState, speed_temporal_prescale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_PRESCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ps",
    },
    {
        .name = "speed_prescale_method",
        .help = "scaling method for the frame, supported options: "
                "[nearest, bilinear, bicubic, lanczos4]",
        .offset = offsetof(SpeedTemporalState, speed_temporal_prescale_method),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_SPEED_PRESCALE_METHOD,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "psm",
    },
    {
        .name = "speed_sigma_nn",
        .help = "standard deviation of neural noise",
        .offset = offsetof(SpeedTemporalState, speed_temporal_sigma_nn),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_SIGMA_NN,
        .min = 0.1,
        .max = 2.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "snn",
    },
    {
        .name = "speed_nn_floor",
        .help = "neural noise floor, expressed in percentage of sigma_nn",
        .offset = offsetof(SpeedTemporalState, speed_temporal_nn_floor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_NN_FLOOR,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "nnf",
    },
    {
        .name = "speed_max_val",
        .help = "maximum value allowed; larger values will be clipped to this "
                "value",
        .offset = offsetof(SpeedTemporalState, speed_temporal_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mxv",
    },
    {
        .name = "speed_use_ref_diff",
        .help = "debug mode: enable additional output",
        .offset = offsetof(SpeedTemporalState, speed_temporal_use_ref_diff),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "urd",
    },
    { 0 }
};

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void)pix_fmt;
    (void)bpc;

    SpeedTemporalState *s = fex->priv;
    s->speed_options = (SpeedOptions) {
        .speed_kernelscale = s->speed_temporal_kernelscale,
        .speed_prescale = s->speed_temporal_prescale,
        .speed_prescale_method = s->speed_temporal_prescale_method,
        .speed_sigma_nn = s->speed_temporal_sigma_nn,
        .speed_nn_floor = s->speed_temporal_nn_floor,
    };

    speed_init(&s->speed_state, &s->speed_options, w, h);

    size_t float_stride = s->speed_state.float_stride;
    size_t frame_size = float_stride * h;
    s->frame_buffer_ref[0] = aligned_malloc(frame_size, 32);
    s->frame_buffer_ref[1] = aligned_malloc(frame_size, 32);
    s->frame_buffer_dis[0] = aligned_malloc(frame_size, 32);
    s->frame_buffer_dis[1] = aligned_malloc(frame_size, 32);

    if (!s->frame_buffer_ref[0] || !s->frame_buffer_ref[1] ||
        !s->frame_buffer_dis[0] || !s->frame_buffer_dis[1])
    {
        return -ENOMEM;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    SpeedTemporalState *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;

    s->index = index;
    int cyclic_index = index % 2;
    int other_index = (index + 1) % 2;

    picture_copy(s->frame_buffer_ref[cyclic_index], s->speed_state.float_stride,
                 ref_pic, -128, ref_pic->bpc, 0);
    picture_copy(s->frame_buffer_dis[cyclic_index], s->speed_state.float_stride,
                 dist_pic, -128, ref_pic->bpc, 0);

    if (index == 0) {
        err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict,
            "Speed_temporal_feature_speed_temporal_score", 0.0, index);
        return err;
    }

    int w = s->speed_state.dimensions.original_width;
    int h = s->speed_state.dimensions.original_height;
    int float_stride = s->speed_state.float_stride;
    subtract_image(s->frame_buffer_ref[other_index],
                   s->frame_buffer_ref[cyclic_index], w, h, float_stride);
    if (s->speed_temporal_use_ref_diff) {
        subtract_image(s->frame_buffer_dis[other_index],
                       s->frame_buffer_ref[cyclic_index], w, h, float_stride);
    } else {
        subtract_image(s->frame_buffer_dis[other_index],
                       s->frame_buffer_dis[cyclic_index], w, h, float_stride);
    }
    float score;
    speed_extract_score(&s->speed_state, &s->speed_options,
                        s->frame_buffer_ref[other_index],
                        s->frame_buffer_dis[other_index], &score);

    err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict,
        "Speed_temporal_feature_speed_temporal_score", score, index);

    if (err) return err;
    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    SpeedTemporalState *s = fex->priv;
    speed_close(&s->speed_state);

    if (s->frame_buffer_ref[0]) aligned_free(s->frame_buffer_ref[0]);
    if (s->frame_buffer_ref[1]) aligned_free(s->frame_buffer_ref[1]);
    if (s->frame_buffer_dis[0]) aligned_free(s->frame_buffer_dis[0]);
    if (s->frame_buffer_dis[1]) aligned_free(s->frame_buffer_dis[1]);

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "Speed_temporal_feature_speed_temporal_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_speed_temporal = {
    .name = "speed_temporal",
    .init = init,
    .extract = extract,
    .options = options,
    .close = close,
    .priv_size = sizeof(SpeedTemporalState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL,
};
