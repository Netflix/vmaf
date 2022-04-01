#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "vidinput.h"

#include "libvmaf/libvmaf.h"

/** Linkage will break without this if using a C++ compiler, and will issue
 * warnings without this for a C compiler*/
#if defined(__cplusplus)
# define OC_EXTERN extern
#else
# define OC_EXTERN
#endif

typedef struct yuv_input {
    FILE *fin;
    unsigned width, height;
    enum VmafPixelFormat pix_fmt;
    unsigned bitdepth;
    size_t dst_buf_sz;
    uint8_t *dst_buf;
    int src_c_dec_v, src_c_dec_h;
    int dst_c_dec_h, dst_c_dec_v;
} yuv_input;


static yuv_input *yuv_input_open(FILE *_fin,
                                 unsigned width, unsigned height,
                                 enum VmafPixelFormat pix_fmt,
                                 unsigned bitdepth)
{
    yuv_input *yuv = malloc(sizeof(*yuv));
    if (!yuv) {
        fprintf(stderr, "Could not allocate yuv reader state.\n");
        return NULL;
    }

    yuv->fin = _fin;
    yuv->width = width;
    yuv->height = height;
    yuv->pix_fmt = pix_fmt;
    yuv->bitdepth = bitdepth;
    bool hbd = yuv->bitdepth > 8;

    switch (yuv->pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        yuv->src_c_dec_h=yuv->dst_c_dec_h=yuv->src_c_dec_v=yuv->dst_c_dec_v=2;
        yuv->dst_buf_sz = (yuv->width*yuv->height
         +2*((yuv->width+1)/2)*((yuv->height+1)/2)) << hbd;
        break;
    case VMAF_PIX_FMT_YUV422P:
        yuv->src_c_dec_h=yuv->dst_c_dec_h=2;
        yuv->src_c_dec_v=yuv->dst_c_dec_v=1;
        yuv->dst_buf_sz = (yuv->width*yuv->height +
            2*(((yuv->width+1)/2) * yuv->height)) << hbd;
        break;
    case VMAF_PIX_FMT_YUV444P:
        yuv->src_c_dec_h=yuv->dst_c_dec_h=yuv->src_c_dec_v=yuv->dst_c_dec_v=1;
        yuv->dst_buf_sz = (yuv->width*yuv->height*3) << hbd;
        break;
    default:
        goto fail; 
    }

    yuv->dst_buf = malloc(yuv->dst_buf_sz);
    if (!yuv->dst_buf) {
        fprintf(stderr, "Could not allocate yuv reader buffer.\n");
        goto fail;
    }

    return yuv;

fail:
    free(yuv);
    return NULL;
}

static int pix_fmt_map(enum VmafPixelFormat pix_fmt)
{
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV420P:
        return PF_420;
    case VMAF_PIX_FMT_YUV422P:
        return PF_422;
    case VMAF_PIX_FMT_YUV444P:
        return PF_444;
    default:
        return 0;
    }
}

static void yuv_input_get_info(yuv_input *_yuv, video_input_info *_info)
{
    memset(_info, 0, sizeof(*_info));
    _info->frame_w = _info->pic_w = _yuv->width;
    _info->frame_h = _info->pic_h = _yuv->height;
    _info->pixel_fmt = pix_fmt_map(_yuv->pix_fmt);
    _info->depth = _yuv->bitdepth;
}

static int yuv_input_fetch_frame(yuv_input *yuv, FILE *fin,
                                 video_input_ycbcr _ycbcr, char _tag[5])
{
    size_t bytes_read = fread(yuv->dst_buf, 1, yuv->dst_buf_sz, fin); 
    if (bytes_read == 0) return 0;
    if (bytes_read != yuv->dst_buf_sz) {
        fprintf(stderr, "Error reading YUV frame data.\n");
        return -1;
    }
    
    (void) _tag;

    unsigned xstride = (yuv->bitdepth>8) ? 2 : 1;
    ptrdiff_t pic_sz = yuv->width * yuv->height * xstride;
    unsigned frame_c_w = yuv->width/yuv->dst_c_dec_h;
    unsigned frame_c_h = yuv->height/yuv->dst_c_dec_v;
    unsigned c_w = (yuv->width+yuv->dst_c_dec_h-1) / yuv->dst_c_dec_h;
    unsigned c_h = (yuv->height+yuv->dst_c_dec_v-1) / yuv->dst_c_dec_v;
    unsigned c_sz = c_w*c_h*xstride;

    _ycbcr[0].width = yuv->width;
    _ycbcr[0].height = yuv->height;
    _ycbcr[0].stride = yuv->width*xstride;
    _ycbcr[0].data = yuv->dst_buf;
    _ycbcr[1].width = frame_c_w;
    _ycbcr[1].height = frame_c_h;
    _ycbcr[1].stride = c_w*xstride;
    _ycbcr[1].data = yuv->dst_buf + pic_sz;
    _ycbcr[2].width = frame_c_w;
    _ycbcr[2].height = frame_c_h;
    _ycbcr[2].stride = c_w*xstride;
    _ycbcr[2].data = _ycbcr[1].data+c_sz;

    return 1;
}

static void yuv_input_close(yuv_input *_yuv){
  free(_yuv->dst_buf);
}

OC_EXTERN const video_input_vtbl YUV_INPUT_VTBL={
  (raw_input_open_func)yuv_input_open,
  (video_input_open_func)NULL,
  (video_input_get_info_func)yuv_input_get_info,
  (video_input_fetch_frame_func)yuv_input_fetch_frame,
  (video_input_close_func)yuv_input_close
};
