/**
 *
 *  Copyright (C) 2022 Intel Corporation.
 *  Copyright (c) 2022-2024 Meta, Inc.
 *
 *     Licensed under the BSD 3-Clause License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/license/bsd-3-clause
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

// #include "integer_funque_filters.h"
#include "../resizer.h"

void vresize_avx2(const int **src, unsigned char *dst, const short *beta, int width);
#if OPTIMISED_COEFF
void step_avx2(const unsigned char *_src, unsigned char *_dst, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int channels, int ksize, int start, int end, int xmin, int xmax);
void hbd_step_avx2(const unsigned short *_src, unsigned short *_dst, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int channels, int ksize, int start, int end, int xmin, int xmax, int bitdepth);
#else
void step_avx2(const unsigned char *_src, unsigned char *_dst, const int *xofs, const int *yofs, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax);
void hbd_step_avx2(const unsigned short *_src, unsigned short *_dst, const int *xofs, const int *yofs, const short *_alpha, const short *_beta, int iwidth, int iheight, int dwidth, int dheight, int channels, int ksize, int start, int end, int xmin, int xmax, int bitdepth);
#endif
//void hbd_resize(const unsigned short *_src, unsigned short *_dst, int iwidth, int iheight, int dwidth, int dheight, int bitdepth);