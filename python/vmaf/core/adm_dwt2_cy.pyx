# import sys

import numpy as np
cimport numpy as np

from libc.stdlib cimport calloc, free

from vmaf.core.adm_dwt2_tools import ALIGN_CEIL, MAX_ALIGN

ctypedef double np_float

cdef struct adm_dwt_band_t:
    np_float *band_a
    np_float *band_v
    np_float *band_h
    np_float *band_d

cdef extern from "../../../libvmaf/src/feature/adm_tools.c":
    void dwt2_src_indices_filt_s(int **src_ind_y, int **src_ind_x, int w, int h)
    void adm_dwt2_d(const np_float *src, const adm_dwt_band_t *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride)

cdef extern from "../../../libvmaf/src/feature/adm.c":
    char *init_dwt_band_d(adm_dwt_band_t *band, char *data_top, size_t buf_sz_one)

cdef extern from "../../../libvmaf/src/mem.c":
    void *aligned_malloc(size_t size, size_t alignment)
    void aligned_free(void *ptr)

cdef extern from "../../../libvmaf/src/feature/offset.c":
    int offset_image_s(float *buf, float off, int width, int height, int stride)  # TODO: find out why need this _s - doesn't seem to be called but if not, symbol not found error

def adm_dwt2_cy(np.ndarray[np.float64_t, ndim=2, mode='c'] a):

    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] a_buf = np.ascontiguousarray(a, dtype=np.float64)
    cdef np_float *aa = <np_float*> a_buf.data  # aa: curr_ref_scale

    cdef int h = len(a)
    cdef int w = len(a[0])
    cdef int h_new, w_new

    cdef np_float *data_buf = NULL
    cdef char *data_top

    cdef char *ind_buf_y = NULL
    cdef char *ind_buf_x = NULL
    cdef char *buf_y_orig = NULL
    cdef char *buf_x_orig = NULL
    cdef int *ind_y[4]
    cdef int *ind_x[4]

    cdef adm_dwt_band_t aa_dwt2  # aa_dwt2: ref_dwt2

    cdef int curr_ref_stride = w * sizeof(np_float)
    cdef int buf_stride = ALIGN_CEIL(((w + 1) // 2) * sizeof(np_float))
    cdef size_t buf_sz_one = <size_t> buf_stride * ((h + 1) // 2)

    cdef int ind_size_y = ALIGN_CEIL(((h + 1) // 2) * sizeof(int))
    cdef int ind_size_x = ALIGN_CEIL(((w + 1) // 2) * sizeof(int))

    # == # must use calloc to initialize mem to 0: adm_dwt2_s doesn't touch every cell for small w and h ==
    # data_buf = <np_float *> aligned_malloc(buf_sz_one * 4, MAX_ALIGN)
    data_buf = <np_float *> calloc(buf_sz_one * 4, 1)
    if not data_buf:
        free(data_buf)
        aligned_free(buf_y_orig)
        aligned_free(buf_x_orig)
        raise MemoryError
    data_top = <char *>data_buf
    data_top = init_dwt_band_d(&aa_dwt2, data_top, buf_sz_one)

    buf_y_orig = <char *> aligned_malloc(ind_size_y * 4, MAX_ALIGN)
    if not buf_y_orig:
        free(data_buf)
        aligned_free(buf_y_orig)
        aligned_free(buf_x_orig)
        raise MemoryError
    ind_buf_y = <char *>buf_y_orig
    ind_y[0] = <int *> ind_buf_y; ind_buf_y += ind_size_y
    ind_y[1] = <int *> ind_buf_y; ind_buf_y += ind_size_y
    ind_y[2] = <int *> ind_buf_y; ind_buf_y += ind_size_y
    ind_y[3] = <int *> ind_buf_y; ind_buf_y += ind_size_y

    buf_x_orig = <char *> aligned_malloc(ind_size_x * 4, MAX_ALIGN)
    if not buf_x_orig:
        free(data_buf)
        aligned_free(buf_y_orig)
        aligned_free(buf_x_orig)
        raise MemoryError
    ind_buf_x = <char *>buf_x_orig
    ind_x[0] = <int *> ind_buf_x; ind_buf_x += ind_size_x
    ind_x[1] = <int *> ind_buf_x; ind_buf_x += ind_size_x
    ind_x[2] = <int *> ind_buf_x; ind_buf_x += ind_size_x
    ind_x[3] = <int *> ind_buf_x; ind_buf_x += ind_size_x

    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] out_a, out_v, out_h, out_d

    try:
        dwt2_src_indices_filt_s(ind_y, ind_x, w, h)
        adm_dwt2_d(aa, &aa_dwt2, ind_y, ind_x, w, h, curr_ref_stride, buf_stride)

        w_new = (w + 1) // 2
        h_new = (h + 1) // 2

        w_new_strided = ALIGN_CEIL(w_new * sizeof(np_float)) // sizeof(np_float)

        # # # ====== debug ======
        # print("h={}, w={}, aa[0]={}, aa[1]={}, aa[2]={}".format(h, w, aa[0], aa[1], aa[2]))
        # print("sizeof(np_float)={}".format(sizeof(np_float)))
        # print("curr_ref_stride={}, buf_stride={}, buf_sz_one={}".format(curr_ref_stride, buf_stride, buf_sz_one))
        # print("ind_size_y={}, ind_size_x={}".format(ind_size_y, ind_size_x))
        # print("h_new={}, w_new={}".format(h_new, w_new))
        # print("np.std(aa_band.band_a)={}".format(np.std(np.asarray(<np.float64_t[:h_new, :w_new]> aa_dwt2.band_a))))
        # print("np.std(aa_band.band_v)={}".format(np.std(np.asarray(<np.float64_t[:h_new, :w_new]> aa_dwt2.band_v))))
        # print("np.std(aa_band.band_h)={}".format(np.std(np.asarray(<np.float64_t[:h_new, :w_new]> aa_dwt2.band_h))))
        # print("np.std(aa_band.band_d)={}".format(np.std(np.asarray(<np.float64_t[:h_new, :w_new]> aa_dwt2.band_d))))
        # for i in range(99):  # for 11 x 9
        #     sys.stdout.write("{}\t".format(aa_dwt2.band_a[i]))
        #     if i%10 == 0:
        #         sys.stdout.write("\n")
        # sys.stdout.write("\n")

        out_a = np.empty((h_new, w_new)).astype(np.float64)
        out_v = np.empty((h_new, w_new)).astype(np.float64)
        out_h = np.empty((h_new, w_new)).astype(np.float64)
        out_d = np.empty((h_new, w_new)).astype(np.float64)

        out_a[...] = np.asarray(<np.float64_t[:h_new, :w_new_strided]> aa_dwt2.band_a)[:, :w_new]
        out_v[...] = np.asarray(<np.float64_t[:h_new, :w_new_strided]> aa_dwt2.band_v)[:, :w_new]
        out_h[...] = np.asarray(<np.float64_t[:h_new, :w_new_strided]> aa_dwt2.band_h)[:, :w_new]
        out_d[...] = np.asarray(<np.float64_t[:h_new, :w_new_strided]> aa_dwt2.band_d)[:, :w_new]

    finally:
        free(data_buf)
        aligned_free(buf_y_orig)
        aligned_free(buf_x_orig)

    return out_a, out_v, out_h, out_d
