int compute_ms_ssim(const float *ref, const float *cmp, int w, int h,
                    int ref_stride, int cmp_stride, double *score,
                    double* l_scores, double* c_scores, double* s_scores);
