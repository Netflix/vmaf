/**
 *
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

#if OPTIMISED_COEFF
void step_neon(const unsigned char *_src, unsigned char *_dst,
               const short *_alpha, const short *_beta, 
               int iwidth, int iheight, int dwidth, int channels, 
               int ksize, int start, int end, int xmin, int xmax);
#else
void step_neon(const unsigned char *_src, unsigned char *_dst, 
               const int *xofs, const int *yofs, 
               const short *_alpha, const short *_beta, 
               int iwidth, int iheight, int dwidth, int dheight, int channels, 
               int ksize, int start, int end, int xmin, int xmax);
#endif
