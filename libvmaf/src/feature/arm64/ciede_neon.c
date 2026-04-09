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

#include <arm_neon.h>
#include <stdint.h>

#include "ciede_neon.h"

void ciede_preprocess_8_neon(const uint8_t *y_buf, const uint8_t *u_buf,
                              const uint8_t *v_buf, float *out_y, float *out_u,
                              float *out_v, int w)
{
    int j = 0;

    for (; j + 4 <= w; j += 4) {
        /* Load 4 uint8 values from each plane (into lower 32 bits) */
        uint8x8_t y8 = vld1_u8(y_buf + j);
        uint8x8_t u8 = vld1_u8(u_buf + j);
        uint8x8_t v8 = vld1_u8(v_buf + j);

        /* Widen uint8 -> uint16 (low 4 elements) */
        uint16x8_t y16 = vmovl_u8(y8);
        uint16x8_t u16 = vmovl_u8(u8);
        uint16x8_t v16 = vmovl_u8(v8);

        /* Widen uint16 -> uint32 (low 4 elements) */
        uint32x4_t y32 = vmovl_u16(vget_low_u16(y16));
        uint32x4_t u32 = vmovl_u16(vget_low_u16(u16));
        uint32x4_t v32 = vmovl_u16(vget_low_u16(v16));

        /* Convert uint32 -> float32 */
        float32x4_t yf = vcvtq_f32_u32(y32);
        float32x4_t uf = vcvtq_f32_u32(u32);
        float32x4_t vf = vcvtq_f32_u32(v32);

        /* Store 4 floats */
        vst1q_f32(out_y + j, yf);
        vst1q_f32(out_u + j, uf);
        vst1q_f32(out_v + j, vf);
    }

    /* Scalar tail */
    for (; j < w; j++) {
        out_y[j] = (float)y_buf[j];
        out_u[j] = (float)u_buf[j];
        out_v[j] = (float)v_buf[j];
    }
}

void ciede_preprocess_16_neon(const uint16_t *y_buf, const uint16_t *u_buf,
                               const uint16_t *v_buf, float *out_y,
                               float *out_u, float *out_v, int w)
{
    int j = 0;

    for (; j + 4 <= w; j += 4) {
        /* Load 4 uint16 values from each plane */
        uint16x4_t y16 = vld1_u16(y_buf + j);
        uint16x4_t u16 = vld1_u16(u_buf + j);
        uint16x4_t v16 = vld1_u16(v_buf + j);

        /* Widen uint16 -> uint32 */
        uint32x4_t y32 = vmovl_u16(y16);
        uint32x4_t u32 = vmovl_u16(u16);
        uint32x4_t v32 = vmovl_u16(v16);

        /* Convert uint32 -> float32 */
        float32x4_t yf = vcvtq_f32_u32(y32);
        float32x4_t uf = vcvtq_f32_u32(u32);
        float32x4_t vf = vcvtq_f32_u32(v32);

        /* Store 4 floats */
        vst1q_f32(out_y + j, yf);
        vst1q_f32(out_u + j, uf);
        vst1q_f32(out_v + j, vf);
    }

    /* Scalar tail */
    for (; j < w; j++) {
        out_y[j] = (float)y_buf[j];
        out_u[j] = (float)u_buf[j];
        out_v[j] = (float)v_buf[j];
    }
}
