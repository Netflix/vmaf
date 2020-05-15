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

#ifndef INTEGER_ADM_FUNCTION_H_
#define INTEGER_ADM_FUNCTION_H_

typedef void pixel;

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
    size_t ind_size_x, ind_size_y;  //strides size for intermidate buffers
    pixel *data_buf;                //buffer for adm intermidiate data calculations
    pixel *tmp_ref;                 //buffer for adm intermidiate data calculations
    pixel *buf_x_orig;              //buffer for storing imgcoeff values along x.
    pixel *buf_y_orig;              //buffer for storing imgcoeff values along y.
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

typedef struct AdmState {
    size_t integer_stride;
    AdmBuffer buf;
} AdmState;

void integer_compute_adm(VmafPicture *ref_pic, VmafPicture *dis_pic, double *score,
                         double *scores, double border_factor, AdmBuffer buf);

#endif /* INTEGER_ADM_FUNCTION_H_ */
