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

#include <stdint.h>

#include <libvmaf/picture.h>

void picture_copy_hbd(float *dst, VmafPicture *src, int offset)
{
    float *float_data = dst;
    uint16_t *data = src->data[0];

    for (unsigned i = 0; i < src->h[0]; i++) {
        for (unsigned j = 0; j < src->w[0]; j++) {
            float_data[j] = (float) data[j] / 4.0 + offset;
        }
        float_data += src->w[0];
        data += src->stride[0] / 2;
    }
    return;
}

//similar to picture_copy_hbd but for integer
void integer_picture_copy_hbd(int16_t *dst, VmafPicture *src, int offset)
{
    int16_t *integer_data = dst;
    uint16_t *data = src->data[0];

    for (unsigned i = 0; i < src->h[0]; i++) {
        for (unsigned j = 0; j < src->w[0]; j++) {
            integer_data[j] = (int16_t)data[j] + offset;
        }
        integer_data += src->w[0];
        data += src->stride[0] / 2;
    }
    return;
}

void picture_copy(float *dst, VmafPicture *src, int offset, unsigned bpc)
{
    if (bpc > 8)
        return picture_copy_hbd(dst, src, offset);

    float *float_data = dst;
    uint8_t *data = src->data[0];

    for (unsigned i = 0; i < src->h[0]; i++) {
        for (unsigned j = 0; j < src->w[0]; j++) {
            float_data[j] = (float) data[j] + offset;
        }
        float_data += src->w[0];
        data += src->stride[0];
    }

    return;
}

//similar to picture_copy but for integer
void integer_picture_copy(int16_t *dst, VmafPicture *src, int offset, unsigned bpc)
{
    if (bpc > 8)
        return integer_picture_copy_hbd(dst, src, offset);

    int16_t *integer_data = dst;
    uint8_t *data = src->data[0];

    for (unsigned i = 0; i < src->h[0]; i++) {
        for (unsigned j = 0; j < src->w[0]; j++) {
            integer_data[j] = (int16_t)data[j] + offset;
        }
        integer_data += src->w[0];
        data += src->stride[0];
    }

    return;
}
