/*
Copyright 2001-2012 Xiph.Org and contributors.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <math.h>
#include <string.h>

#include "feature_collector.h"
#include "feature_extractor.h"

#define KERNEL_SHIFT (8)
#define KERNEL_WEIGHT (1<<KERNEL_SHIFT)
#define KERNEL_ROUND ((1<<KERNEL_SHIFT)>>1)

#ifndef M_PI
#define M_PI 3.141592653589793238462643
#endif

static int gaussian_filter_init(unsigned **_kernel,double _sigma,int _max_len){
  unsigned *kernel;
  double    scale;
  double    nhisigma2;
  double    s;
  double    len;
  unsigned  sum;
  int       kernel_len;
  int       kernel_sz;
  int       ci;
  scale=1/(sqrt(2*M_PI)*_sigma);
  nhisigma2=-0.5/(_sigma*_sigma);
  /*Compute the kernel size so that the error in the first truncated
     coefficient is no larger than 0.5*KERNEL_WEIGHT.
    There is no point in going beyond this given our working precision.*/
  s=sqrt(0.5*M_PI)*_sigma*(1.0/KERNEL_WEIGHT);
  if(s>=1)len=0;
  else len=floor(_sigma*sqrt(-2*log(s)));
  kernel_len=len>=_max_len?_max_len-1:(int)len;
  kernel_sz=kernel_len<<1|1;
  kernel=(unsigned *)malloc(kernel_sz*sizeof(*kernel));
  sum=0;
  for(ci=kernel_len;ci>0;ci--){
    kernel[kernel_len-ci]=kernel[kernel_len+ci]=
     (unsigned)(KERNEL_WEIGHT*scale*exp(nhisigma2*ci*ci)+0.5);
    sum+=kernel[kernel_len-ci];
  }
  kernel[kernel_len]=KERNEL_WEIGHT-(sum<<1);
  *_kernel=kernel;
  return kernel_sz;
}

typedef struct ssim_moments ssim_moments;

struct ssim_moments{
  int64_t mux;
  int64_t muy;
  int64_t x2;
  int64_t xy;
  int64_t y2;
  int64_t w;
};

#define SSIM_K1 (0.01*0.01)
#define SSIM_K2 (0.03*0.03)

static double calc_ssim(const unsigned char *_src,int _systride,
 const unsigned char *_dst,int _dystride,double _par,int depth,int _w,int _h){
  ssim_moments  *line_buf;
  ssim_moments **lines;
  double         ssim;
  double         ssimw;
  unsigned      *hkernel;
  int            hkernel_sz;
  int            hkernel_offs;
  unsigned      *vkernel;
  int            vkernel_sz;
  int            vkernel_offs;
  int            log_line_sz;
  int            line_sz;
  int            line_mask;
  int            x;
  int            y;
  int            samplemax;
  samplemax = (1 << depth) - 1;
  vkernel_sz=gaussian_filter_init(&vkernel,1.5,5);
  vkernel_offs=vkernel_sz>>1;
  for(line_sz=1,log_line_sz=0;line_sz<vkernel_sz;line_sz<<=1,log_line_sz++);
  line_mask=line_sz-1;
  lines=(ssim_moments **)malloc(line_sz*sizeof(*lines));
  lines[0]=line_buf=(ssim_moments *)malloc(line_sz*_w*sizeof(*line_buf));
  for(y=1;y<line_sz;y++)lines[y]=lines[y-1]+_w;
  hkernel_sz=gaussian_filter_init(&hkernel,1.5,5);
  hkernel_offs=hkernel_sz>>1;
  ssim=0;
  ssimw=0;
  for(y=0;y<_h+vkernel_offs;y++){
    ssim_moments *buf;
    int           k;
    int           k_min;
    int           k_max;
    if(y<_h){
      buf=lines[y&line_mask];
      for(x=0;x<_w;x++){
        ssim_moments m;
        memset(&m,0,sizeof(m));
        k_min=hkernel_offs-x<=0?0:hkernel_offs-x;
        k_max=x+hkernel_offs-_w+1<=0?
         hkernel_sz:hkernel_sz-(x+hkernel_offs-_w+1);
        for(k=k_min;k<k_max;k++){
          signed s;
          signed d;
          signed window;
          if (depth > 8) {
            s = _src[(x-hkernel_offs+k)*2] +
             (_src[(x-hkernel_offs+k)*2 + 1] << 8);
            d = _dst[(x-hkernel_offs+k)*2] +
             (_dst[(x-hkernel_offs+k)*2 + 1] << 8);
          } else {
            s=_src[(x-hkernel_offs+k)];
            d=_dst[(x-hkernel_offs+k)];
          }
          window=hkernel[k];
          m.mux+=window*s;
          m.muy+=window*d;
          m.x2+=window*s*s;
          m.xy+=window*s*d;
          m.y2+=window*d*d;
          m.w+=window;
        }
        *(buf+x)=*&m;
      }
      _src+=_systride;
      _dst+=_dystride;
    }
    if(y>=vkernel_offs){
      k_min=vkernel_sz-y-1<=0?0:vkernel_sz-y-1;
      k_max=y+1-_h<=0?vkernel_sz:vkernel_sz-(y+1-_h);
      for(x=0;x<_w;x++){
        ssim_moments m;
        double       c1;
        double       c2;
        double       mx2;
        double       mxy;
        double       my2;
        double       w;
        memset(&m,0,sizeof(m));
        for(k=k_min;k<k_max;k++){
          signed window;
          buf = lines[(y + 1 - vkernel_sz + k) & line_mask] + x;
          window=vkernel[k];
          m.mux+=window*buf->mux;
          m.muy+=window*buf->muy;
          m.x2+=window*buf->x2;
          m.xy+=window*buf->xy;
          m.y2+=window*buf->y2;
          m.w+=window*buf->w;
        }
        w=m.w;
        c1=samplemax*samplemax*SSIM_K1*w*w;
        c2=samplemax*samplemax*SSIM_K2*w*w;
        mx2=m.mux*(double)m.mux;
        mxy=m.mux*(double)m.muy;
        my2=m.muy*(double)m.muy;
        ssim+=m.w*(2*mxy+c1)*(c2+2*(m.xy*w-mxy))/
         ((mx2+my2+c1)*(m.x2*w-mx2+m.y2*w-my2+c2));
        ssimw+=m.w;
      }
    }
  }
  free(line_buf);
  free(lines);
  free(vkernel);
  free(hkernel);
  return ssim/ssimw;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    (void) ref_pic_90;
    (void) dist_pic_90;

    double score =
        calc_ssim(ref_pic->data[0], ref_pic->stride[0],
                  dist_pic->data[0], dist_pic->stride[0], 1.0, ref_pic->bpc,
                  ref_pic->w[0], ref_pic->h[0]);
    int err =
        vmaf_feature_collector_append(feature_collector, "ssim", score, index);
    if (err) return err;
    return 0;
}

static int close(VmafFeatureExtractor *fex)
{
    return 0;
}

static const char *provided_features[] = {
    "ssim",
    NULL
};

VmafFeatureExtractor vmaf_fex_ssim = {
    .name = "ssim",
    .init = init,
    .extract = extract,
    .close = close,
    .provided_features = provided_features,
};
