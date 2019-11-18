#ifndef __VMAF_PICTURE_H__
#define __VMAF_PICTURE_H__

#include <stdatomic.h>
#include <stddef.h>

enum VmafPixelFormat {
    VMAF_PIX_FMT_UNKNOWN,
    VMAF_PIX_FMT_YUV420P,
    VMAF_PIX_FMT_YUV422P,
    VMAF_PIX_FMT_YUV444P,
};

typedef void pixel;

typedef struct {
    enum VmafPixelFormat pix_fmt;
    unsigned bpc;
    unsigned w[3], h[3];
    ptrdiff_t stride[3];
    pixel *data[3];
    atomic_int *ref_cnt;
} VmafPicture;

int vmaf_picture_alloc(VmafPicture *pic, enum VmafPixelFormat pix_fmt,
                       unsigned bpc, unsigned w, unsigned h);

int vmaf_picture_unref(VmafPicture *pic);

#endif /* __VMAF_PICTURE_H__ */
