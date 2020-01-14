int compute_psnr(const float *ref, const float *dis, int w, int h,
                 int ref_stride, int dis_stride,
                 double *score, double peak, double psnr_max);
