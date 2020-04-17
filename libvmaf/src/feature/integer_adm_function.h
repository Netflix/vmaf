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

typedef struct Integer_AdmState {
    size_t integer_stride, ind_size_x, ind_size_y, buf_sz_one;  //strides size for intermidate buffers
    pixel *data_buf;    //buffer for adm intermidiate data calculations
    pixel *tmp_ref;     //buffer for adm intermidiate data calculations
    pixel *buf_x_orig;  //buffer for storing imgcoeff values to reduces the control code cycles along x.
    pixel *buf_y_orig;  //buffer for storing imgcoeff values to reduces the control code cycles along y.
} Integer_AdmState;

int integer_compute_adm(const pixel *ref, const pixel *dis, int w, int h,
    int ref_stride, int dis_stride, double *score, double *score_num,
    double *score_den, double *scores, double border_factor, int inp_size_bits, Integer_AdmState *s);

#endif /* INTEGER_ADM_FUNCTION_H_ */
