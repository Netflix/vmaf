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
