import pywt

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

cdef struct adm_dwt_band_t_d:
    double *band_a
    double *band_v
    double *band_h
    double *band_d

cdef extern from "../../../libvmaf/src/feature/adm_tools.c":
    void dwt2_src_indices_filt_s(int **src_ind_y, int **src_ind_x, int w, int h);
    void adm_dwt2_d(const double *src, const adm_dwt_band_t_d *dst, int **ind_y, int **ind_x, int w, int h, int src_stride, int dst_stride)

cdef extern from "../../../libvmaf/src/mem.c":
    void aligned_free(void *ptr)

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
    cdef double *aa = <double*> a_buf.data

    cdef int h = len(a)
    cdef int w = len(a[0])

    cdef int *ind_y[4]
    cdef int *ind_x[4]

    cdef int ind_size_y = ALIGN_CEIL(((h + 1) / 2) * sizeof(int))
    cdef int ind_size_x = ALIGN_CEIL(((w + 1) / 2) * sizeof(int))

    cdef int *ind_y_mem = <int *>malloc(ind_size_y * 4)  # TODO: combine allocating ind_x_mem and ind_y_mem
    if not ind_y_mem:
        raise MemoryError

    cdef int *ind_x_mem = <int *>malloc(ind_size_x * 4)
    if not ind_x_mem:
        free(ind_y_mem)
        raise MemoryError

    cdef adm_dwt_band_t_d aa_band

    cdef int curr_ref_stride = h * sizeof(double)
    cdef int buf_stride = ALIGN_CEIL(((w + 1) / 2) * sizeof(double))

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

        dwt2_src_indices_filt_s(ind_y, ind_x, w, h)

        # adm_dwt2_d(a, &aa_band, ind_y, ind_x, w, h, curr_ref_stride, buf_stride)

        print("h={}, w={}, aa[0]={}, aa[1]={}, aa[2]={}".format(h, w, aa[0], aa[1], aa[2]))
        print("ind_size_y={}, ind_size_x={}".format(ind_size_y, ind_size_x))
        print("ind_y[0]: {}, {}, {}, {}, {}".format(ind_y[0][0], ind_y[0][1], ind_y[0][2], ind_y[0][3], ind_y[0][4]))
        print("ind_y[1]: {}, {}, {}, {}, {}".format(ind_y[1][0], ind_y[1][1], ind_y[1][2], ind_y[1][3], ind_y[1][4]))
        print("ind_y[2]: {}, {}, {}, {}, {}".format(ind_y[2][0], ind_y[2][1], ind_y[2][2], ind_y[2][3], ind_y[2][4]))
        print("ind_y[3]: {}, {}, {}, {}, {}".format(ind_y[3][0], ind_y[3][1], ind_y[3][2], ind_y[3][3], ind_y[3][4]))
        print("ind_x[0]: {}, {}, {}, {}, {}".format(ind_x[0][0], ind_x[0][1], ind_x[0][2], ind_x[0][3], ind_x[0][4]))
        print("ind_x[1]: {}, {}, {}, {}, {}".format(ind_x[1][0], ind_x[1][1], ind_x[1][2], ind_x[1][3], ind_x[1][4]))
        print("ind_x[2]: {}, {}, {}, {}, {}".format(ind_x[2][0], ind_x[2][1], ind_x[2][2], ind_x[2][3], ind_x[2][4]))
        print("ind_x[3]: {}, {}, {}, {}, {}".format(ind_x[3][0], ind_x[3][1], ind_x[3][2], ind_x[3][3], ind_x[3][4]))
        print("curr_ref_stride={}, buf_stride={}".format(curr_ref_stride, buf_stride))  # FIXME


    finally:
        free(ind_y_mem)
        free(ind_x_mem)


    a, ds = pywt.dwt2(a, 'db2', 'periodization')


    # print("order_spam(5) prints:")
    # order_spam(5)
    #
    # cdef int x = parse_charptr_to_py_int('5')
    # print("parse_charptr_to_py_int('5') returns {} of type {}".format(x, type(x)))
    #
    # print("f(3) = {}".format(f(3)))

    return a, ds
