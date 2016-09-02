import numpy as np
import scipy.misc
import scipy.ndimage

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"


def _gauss_window(lw, sigma):
    sd = float(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def _hp_image(image):
    extend_mode = 'reflect'
    image = np.array(image).astype(np.float32)
    w, h = image.shape
    mu_image = np.zeros((w, h))
    _avg_window = _gauss_window(3, 1.0)
    scipy.ndimage.correlate1d(image, _avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, _avg_window, 1, mu_image, mode=extend_mode)
    return image - mu_image


def _var_image(hpimg):
    extend_mode = 'reflect'
    w, h = hpimg.shape
    varimg = np.zeros((w, h))
    _var_window = _gauss_window(3, 1.0)
    scipy.ndimage.correlate1d(hpimg**2, _var_window, 0, varimg, mode=extend_mode)
    scipy.ndimage.correlate1d(varimg, _var_window, 1, varimg, mode=extend_mode)
    return varimg


def as_one_hot(label_list):
    return np.eye(2)[np.array(label_list).astype(np.int)]


def create_hp_yuv_4channel(yuvimg):
    yuvimg = yuvimg.astype(np.float32)
    yuvimg /= 255.0
    hp_y = _hp_image(yuvimg[:, :, 0])
    hp_u = _hp_image(yuvimg[:, :, 1])
    hp_v = _hp_image(yuvimg[:, :, 2])
    sigma = np.sqrt(_var_image(hp_y))

    # stack together to make 4 channel image
    return np.dstack((hp_y, hp_u, hp_v, sigma))


def dstack_y_u_v(y, u, v):
    # make y, u, v consistent in size
    if u.shape != y.shape:
        u = scipy.misc.imresize(u, size=y.shape, interp='bicubic')
    if v.shape != y.shape:
        v = scipy.misc.imresize(v, size=y.shape, interp='bicubic')
    return np.dstack((y, u, v))