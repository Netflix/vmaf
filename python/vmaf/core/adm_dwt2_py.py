import pywt


def adm_dwt2_py(a):
    a, ds = pywt.dwt2(a, 'db2', 'periodization')
    h, v, d = ds
    return a, v, h, d


def adm_idwt2_py(a_v_h_d):
    a, v, h, d = a_v_h_d
    ds = (h, v, d)
    a = pywt.idwt2((a, ds), 'db2', 'periodization')
    return a
