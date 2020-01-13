int compute_ssim(const float *ref, const float *cmp, int w, int h,
                 int ref_stride, int cmp_stride, double *score,
                 double *l_score, double *c_score, double *s_score);
