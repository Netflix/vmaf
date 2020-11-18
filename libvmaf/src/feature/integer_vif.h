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

#ifndef FEATURE_VIF_H_
#define FEATURE_VIF_H_

#include <stdint.h>

static const uint16_t vif_filter1d_table[4][18] = {
    { 489, 935, 1640, 2640, 3896, 5274, 6547, 7455, 7784, 7455, 6547, 5274, 3896, 2640, 1640, 935, 489, 0 },
    { 1244, 3663, 7925, 12590, 14692, 12590, 7925, 3663, 1244, 0 },
    { 3571, 16004, 26386, 16004, 3571, 0 },
    { 10904, 43728, 10904, 0 }
};

static const int vif_filter1d_width[4] = { 17, 9, 5, 3 };

typedef struct VifBuffer {
    void *data;

    void *ref;
    void *dis;
    uint16_t *mu1;
    uint16_t *mu2;
    uint32_t *mu1_32;
    uint32_t *mu2_32;
    uint32_t *ref_sq;
    uint32_t *dis_sq;
    uint32_t *ref_dis;

    struct {
        uint32_t *mu1;
        uint32_t *mu2;
        uint32_t *ref;
        uint32_t *dis;
        uint32_t *ref_dis;
        uint32_t *ref_convol;
        uint32_t *dis_convol;
    } tmp;

    ptrdiff_t stride;
    ptrdiff_t stride_16;
    ptrdiff_t stride_32;
    ptrdiff_t stride_tmp;
} VifBuffer;

static inline void PADDING_SQ_DATA(VifBuffer buf, int w, unsigned fwidth_half)
{
    for (unsigned f = 1; f <= fwidth_half; ++f) {
        int left_point = -f;
        int right_point = f;
        buf.tmp.mu1[left_point] = buf.tmp.mu1[right_point];
        buf.tmp.mu2[left_point] = buf.tmp.mu2[right_point];
        buf.tmp.ref[left_point] = buf.tmp.ref[right_point];
        buf.tmp.dis[left_point] = buf.tmp.dis[right_point];
        buf.tmp.ref_dis[left_point] = buf.tmp.ref_dis[right_point];

        left_point = w - 1 - f;
        right_point = w - 1 + f;
        buf.tmp.mu1[right_point] = buf.tmp.mu1[left_point];
        buf.tmp.mu2[right_point] = buf.tmp.mu2[left_point];
        buf.tmp.ref[right_point] = buf.tmp.ref[left_point];
        buf.tmp.dis[right_point] = buf.tmp.dis[left_point];
        buf.tmp.ref_dis[right_point] = buf.tmp.ref_dis[left_point];
    }
}

static inline void PADDING_SQ_DATA_2(VifBuffer buf, int w, unsigned fwidth_half)
{
    for (unsigned f = 1; f <= fwidth_half; ++f) {
        int left_point = -f;
        int right_point = f;
        buf.tmp.ref_convol[left_point] = buf.tmp.ref_convol[right_point];
        buf.tmp.dis_convol[left_point] = buf.tmp.dis_convol[right_point];

        left_point = w - 1 - f;
        right_point = w - 1 + f;
        buf.tmp.ref_convol[right_point] = buf.tmp.ref_convol[left_point];
        buf.tmp.dis_convol[right_point] = buf.tmp.dis_convol[left_point];
    }
}

#endif /* _FEATURE_VIF_H_ */
