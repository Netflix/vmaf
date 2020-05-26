import pywt

cimport numpy as np

def adm_dwt2_cy(np.ndarray[np.float_t, ndim=2] a):
    a, ds = pywt.dwt2(a, 'db2', 'periodization')
    return a, ds
