# import pywt

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

cdef struct adm_dwt_band_t_d:
    double *band_a
    double *band_v
    double *band_h
    double *band_d

cdef extern from "../../../libvmaf/src/feature/adm_tools.c":
    void dwt2_src_indices_filt_s(int **src_ind_y, int **src_ind_x, int w, int h)
    void adm_dwt2_d(const double *src, const adm_dwt_band_t_d *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride)

cdef extern from "../../../libvmaf/src/feature/adm.c":
    char *init_dwt_band_d(adm_dwt_band_t_d *band, char *data_top, size_t buf_sz_one)

cdef extern from "../../../libvmaf/src/mem.c":
    void aligned_free(void *ptr)

cdef extern from "../../../libvmaf/src/feature/offset.c":
    int offset_image_s(float *buf, float off, int width, int height, int stride)  # TODO: find out why need this _s - doesn't seem to be called but if not, symbol not found error

# cdef extern from "spam.c":
#     void order_spam(int tons)
#
# from libc.stdlib cimport atoi
# cdef parse_charptr_to_py_int(char* s):
#     assert s is not NULL, "byte string value is NULL"
#     return atoi(s)  # note: atoi() has no error detection!
#
# from libc.math cimport sin
# cdef double f(double x):
#     return sin(x * x)

ctypedef double np_float

MAX_ALIGN = 32

def ALIGN_CEIL(x):
    # ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
    if x % MAX_ALIGN != 0:
        y = MAX_ALIGN - x % MAX_ALIGN
    else:
        y = 0
    return x + y

def adm_dwt2_cy(np.ndarray[np.float_t, ndim=2, mode='c'] a):

    # cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
    # cdef unsigned int* im_buff = <unsigned int*> np_buff.data

    # cdef np.ndarray[double, ndim=2, mode="c"] a_cython = np.asarray(a, dtype = float, order="C")
    # cdef double** point_to_a = <double **>malloc(N * sizeof(double*))
    # if not point_to_a: raise MemoryError
    # try:
    #     for i in range(N):
    #         point_to_a[i] = &a_cython[i, 0]
    #     # Call the C function that expects a double**
    #     myfunc(... &point_to_a[0], ...)
    # finally:
    #     free(point_to_a)

    cdef np.ndarray[np.float_t, ndim=2, mode='c'] a_buf = np.ascontiguousarray(a, dtype=np.float)
    cdef np_float *aa = <np_float*> a_buf.data

    cdef int h = len(a)
    cdef int w = len(a[0])

    cdef int curr_ref_stride = w * sizeof(np_float)
    cdef int buf_stride = ALIGN_CEIL(((w + 1) // 2) * sizeof(np_float))
    cdef size_t buf_sz_one = <size_t> buf_stride * ((h + 1) // 2)

    cdef int ind_size_y = ALIGN_CEIL(((h + 1) // 2) * sizeof(int))
    cdef int ind_size_x = ALIGN_CEIL(((w + 1) // 2) * sizeof(int))

    cdef int *ind_y_mem = <int *>malloc(ind_size_y * 4)  # TODO: combine allocating ind_x_mem and ind_y_mem and data_mem
    if not ind_y_mem:
        raise MemoryError

    cdef int *ind_x_mem = <int *>malloc(ind_size_x * 4)
    if not ind_x_mem:
        free(ind_y_mem)
        raise MemoryError

    cdef char * data_mem = <char *>malloc(buf_sz_one * 16)   # FIXME: supposed to be * 4, but resulting in corrupted data
    if not data_mem:
        free(ind_y_mem)
        free(ind_x_mem)
        raise MemoryError

    cdef int *ind_y[4]
    cdef int *ind_x[4]
    cdef adm_dwt_band_t_d aa_band

    cdef int h_new = (h + 1) // 2
    cdef int w_new = (w + 1) // 2

    cdef np.ndarray[np.float_t, ndim=2, mode='c'] a_new, ds_h, ds_v, ds_d

    cdef char *ptr
    try:
        ptr = <char *>ind_y_mem
        ind_y[0] = <int *> ptr; ptr += ind_size_y
        ind_y[1] = <int *> ptr; ptr += ind_size_y
        ind_y[2] = <int *> ptr; ptr += ind_size_y
        ind_y[3] = <int *> ptr

        ptr = <char *>ind_x_mem
        ind_x[0] = <int *> ptr; ptr += ind_size_x
        ind_x[1] = <int *> ptr; ptr += ind_size_x
        ind_x[2] = <int *> ptr; ptr += ind_size_x
        ind_x[3] = <int *> ptr

        ptr = <char *>data_mem
        ptr = init_dwt_band_d(&aa_band, ptr, buf_sz_one)

        dwt2_src_indices_filt_s(ind_y, ind_x, w, h)

        adm_dwt2_d(aa, &aa_band, ind_y, ind_x, w, h, curr_ref_stride, buf_stride)

        # # ====== debug ======
        # print("h={}, w={}, aa[0]={}, aa[1]={}, aa[2]={}".format(h, w, aa[0], aa[1], aa[2]))
        # print("ind_size_y={}, ind_size_x={}".format(ind_size_y, ind_size_x))
        # print("ind_y[0]: {}, {}, {}, {}, {}".format(ind_y[0][0], ind_y[0][1], ind_y[0][2], ind_y[0][3], ind_y[0][4]))
        # print("ind_y[1]: {}, {}, {}, {}, {}".format(ind_y[1][0], ind_y[1][1], ind_y[1][2], ind_y[1][3], ind_y[1][4]))
        # print("ind_y[2]: {}, {}, {}, {}, {}".format(ind_y[2][0], ind_y[2][1], ind_y[2][2], ind_y[2][3], ind_y[2][4]))
        # print("ind_y[3]: {}, {}, {}, {}, {}".format(ind_y[3][0], ind_y[3][1], ind_y[3][2], ind_y[3][3], ind_y[3][4]))
        # print("ind_x[0]: {}, {}, {}, {}, {}".format(ind_x[0][0], ind_x[0][1], ind_x[0][2], ind_x[0][3], ind_x[0][4]))
        # print("ind_x[1]: {}, {}, {}, {}, {}".format(ind_x[1][0], ind_x[1][1], ind_x[1][2], ind_x[1][3], ind_x[1][4]))
        # print("ind_x[2]: {}, {}, {}, {}, {}".format(ind_x[2][0], ind_x[2][1], ind_x[2][2], ind_x[2][3], ind_x[2][4]))
        # print("ind_x[3]: {}, {}, {}, {}, {}".format(ind_x[3][0], ind_x[3][1], ind_x[3][2], ind_x[3][3], ind_x[3][4]))
        # print("curr_ref_stride={}, buf_stride={}, buf_sz_one={}".format(curr_ref_stride, buf_stride, buf_sz_one))
        # print("aa_band->band_a: {}, {}, {}, {}, {}".format(aa_band.band_a[0], aa_band.band_a[1], aa_band.band_a[2], aa_band.band_a[3], aa_band.band_a[4]))
        # print("aa_band->band_v: {}, {}, {}, {}, {}".format(aa_band.band_v[0], aa_band.band_v[1], aa_band.band_v[2], aa_band.band_v[3], aa_band.band_v[4]))
        # print("aa_band->band_h: {}, {}, {}, {}, {}".format(aa_band.band_h[0], aa_band.band_h[1], aa_band.band_h[2], aa_band.band_h[3], aa_band.band_h[4]))
        # print("aa_band->band_d: {}, {}, {}, {}, {}".format(aa_band.band_d[0], aa_band.band_d[1], aa_band.band_d[2], aa_band.band_d[3], aa_band.band_d[4]))
        # print("np.mean(aa_band.band_a)={}".format(np.mean(np.asarray(<np.float_t[:h_new, :w_new]> aa_band.band_a))))
        # print("np.mean(aa_band.band_v)={}".format(np.mean(np.asarray(<np.float_t[:h_new, :w_new]> aa_band.band_v))))
        # print("np.mean(aa_band.band_h)={}".format(np.mean(np.asarray(<np.float_t[:h_new, :w_new]> aa_band.band_h))))
        # print("np.mean(aa_band.band_d)={}".format(np.mean(np.asarray(<np.float_t[:h_new, :w_new]> aa_band.band_d))))

        a_new = np.ones((h_new, w_new))
        ds_h  = np.ones((h_new, w_new))
        ds_v  = np.ones((h_new, w_new))
        ds_d  = np.ones((h_new, w_new))

        a_new[...] = np.asarray(<np.float_t[:h_new, :w_new]> aa_band.band_a)[...]
        ds_h[...]  = np.asarray(<np.float_t[:h_new, :w_new]> aa_band.band_h)[...]
        ds_v[...]  = np.asarray(<np.float_t[:h_new, :w_new]> aa_band.band_v)[...]
        ds_d[...]  = np.asarray(<np.float_t[:h_new, :w_new]> aa_band.band_d)[...]

    finally:
        free(ind_y_mem)
        free(ind_x_mem)
        free(data_mem)

    # print("order_spam(5) prints:")
    # order_spam(5)
    #
    # cdef int x = parse_charptr_to_py_int('5')
    # print("parse_charptr_to_py_int('5') returns {} of type {}".format(x, type(x)))
    #
    # print("f(3) = {}".format(f(3)))

    # a, ds = pywt.dwt2(a, 'db2', 'periodization')

    # return a, ds
    return a_new, [ds_h, ds_v, ds_d]