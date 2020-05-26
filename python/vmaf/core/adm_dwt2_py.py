import pywt


def adm_dwt2_py(a):
    a, ds = pywt.dwt2(a, 'db2', 'periodization')
    return a, ds
