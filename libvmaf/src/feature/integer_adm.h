#ifndef FEATURE_ADM_H_
#define FEATURE_ADM_H_

#include "mem.h"
#include "stdio.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int32_t div_lookup[65537];
static const int32_t div_Q_factor = 1073741824; // 2^30

static inline void div_lookup_generator() {
    for (int i = 1; i <= 32768; ++i) {
        int32_t recip = (int32_t)(div_Q_factor / i);
        div_lookup[32768 + i] = recip;
        div_lookup[32768 - i] = 0 - recip;
    }
}

typedef struct adm_dwt_band_t {
    int16_t *band_a; /* Low-pass V + low-pass H. */
    int16_t *band_v; /* Low-pass V + high-pass H. */
    int16_t *band_h; /* High-pass V + low-pass H. */
    int16_t *band_d; /* High-pass V + high-pass H. */
} adm_dwt_band_t;

typedef struct i4_adm_dwt_band_t {
    int32_t *band_a; /* Low-pass V + low-pass H. */
    int32_t *band_v; /* Low-pass V + high-pass H. */
    int32_t *band_h; /* High-pass V + low-pass H. */
    int32_t *band_d; /* High-pass V + high-pass H. */
} i4_adm_dwt_band_t;

typedef struct AdmBuffer {
    size_t ind_size_x, ind_size_y; // strides size for intermidate buffers
    void *data_buf;   // buffer for adm intermidiate data calculations
    void *tmp_ref;    // buffer for adm intermidiate data calculations
    void *buf_x_orig; // buffer for storing imgcoeff values along x.
    void *buf_y_orig; // buffer for storing imgcoeff values along y.
    int *ind_y[4], *ind_x[4];

    adm_dwt_band_t ref_dwt2;
    adm_dwt_band_t dis_dwt2;
    adm_dwt_band_t decouple_r;
    adm_dwt_band_t decouple_a;
    adm_dwt_band_t csf_a;
    adm_dwt_band_t csf_f;

    i4_adm_dwt_band_t i4_ref_dwt2;
    i4_adm_dwt_band_t i4_dis_dwt2;
    i4_adm_dwt_band_t i4_decouple_r;
    i4_adm_dwt_band_t i4_decouple_a;
    i4_adm_dwt_band_t i4_csf_a;
    i4_adm_dwt_band_t i4_csf_f;
} AdmBuffer;

#ifndef NUM_BUFS_ADM
#define NUM_BUFS_ADM 30
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif // M_PI

/* Enhancement gain imposed on adm, must be >= 1.0, where 1.0 means the gain is completely disabled */
#ifndef DEFAULT_ADM_ENHN_GAIN_LIMIT
#define DEFAULT_ADM_ENHN_GAIN_LIMIT (100.0)
#endif // !DEFAULT_ADM_ENHN_GAIN_LIMIT

#ifndef DEFAULT_ADM_NORM_VIEW_DIST
#define DEFAULT_ADM_NORM_VIEW_DIST (3.0)
#endif // !DEFAULT_ADM_NORM_VIEW_DIST

#ifndef DEFAULT_ADM_REF_DISPLAY_HEIGHT
#define DEFAULT_ADM_REF_DISPLAY_HEIGHT (1080)
#endif // !DEFAULT_ADM_REF_DISPLAY_HEIGHT

#ifndef ADM_BORDER_FACTOR
#define ADM_BORDER_FACTOR (0.1)
#endif // !ADM_BORDER_FACTOR

#define DIVS(n, d) ((n) / (d))

static const int16_t dwt2_db2_coeffs_lo[10] = {15826, 27411, 7345, -4240};
static const int16_t dwt2_db2_coeffs_hi[10] = {-4240, -7345, 27411, -15826};

static const int32_t dwt2_db2_coeffs_lo_sum = 46342;
static const int32_t dwt2_db2_coeffs_hi_sum = 0;

#ifndef ONE_BY_15
#define ONE_BY_15 8738
#endif

#ifndef I4_ONE_BY_15
#define I4_ONE_BY_15 286331153
#endif

/* ================= */
/* Noise floor model */
/* ================= */

/*
 * The following dwt visibility threshold parameters are taken from
 * "Visibility of Wavelet Quantization Noise"
 * by A. B. Watson, G. Y. Yang, J. A. Solomon and J. Villasenor
 * IEEE Trans. on Image Processing, Vol. 6, No 8, Aug. 1997
 * Page 1170, formula (7) and corresponding Table IV
 * Table IV has 2 entries for Cb and Cr thresholds
 * Chose those corresponding to subject "sfl" since they are lower
 * These thresholds were obtained and modeled for the 7-9 biorthogonal wavelet
 * basis
 */

/*
 * The following dwt visibility threshold parameters are taken from
 * "Visibility of Wavelet Quantization Noise"
 * by A. B. Watson, G. Y. Yang, J. A. Solomon and J. Villasenor
 * IEEE Trans. on Image Processing, Vol. 6, No 8, Aug. 1997
 * Page 1170, formula (7) and corresponding Table IV
 * Table IV has 2 entries for Cb and Cr thresholds
 * Chose those corresponding to subject "sfl" since they are lower
 * These thresholds were obtained and modeled for the 7-9 biorthogonal wavelet
 * basis
 */
struct dwt_model_params {
    float a;
    float k;
    float f0;
    float g[4];
};

// 0 -> Y, 1 -> Cb, 2 -> Cr
static const struct dwt_model_params dwt_7_9_YCbCr_threshold[3] = {
    {.a = 0.495, .k = 0.466, .f0 = 0.401, .g = {1.501, 1.0, 0.534, 1.0}},
    {.a = 1.633, .k = 0.353, .f0 = 0.209, .g = {1.520, 1.0, 0.502, 1.0}},
    {.a = 0.944, .k = 0.521, .f0 = 0.404, .g = {1.868, 1.0, 0.516, 1.0}}};

/*
 * The following dwt basis function amplitudes, A(lambda,theta), are taken from
 * "Visibility of Wavelet Quantization Noise"
 * by A. B. Watson, G. Y. Yang, J. A. Solomon and J. Villasenor
 * IEEE Trans. on Image Processing, Vol. 6, No 8, Aug. 1997
 * Page 1172, Table V
 * The table has been transposed, i.e. it can be used directly to obtain
 * A[lambda][theta]
 * These amplitudes were calculated for the 7-9 biorthogonal wavelet basis
 */
static const float dwt_7_9_basis_function_amplitudes[6][4] = {
    {0.62171, 0.67234, 0.72709, 0.67234},
    {0.34537, 0.41317, 0.49428, 0.41317},
    {0.18004, 0.22727, 0.28688, 0.22727},
    {0.091401, 0.11792, 0.15214, 0.11792},
    {0.045943, 0.059758, 0.077727, 0.059758},
    {0.023013, 0.030018, 0.039156, 0.030018}};

#endif /* _FEATURE_ADM_H_ */
