import pywt


def adm_dwt2_py(a):
    a, ds = pywt.dwt2(a, 'db2', 'periodization')
    return a, ds


def adm_idwt2_py(a_ds):
    a = pywt.idwt2(a_ds, 'db2', 'periodization')
    return a
