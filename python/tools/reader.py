__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os

import numpy as np

class YuvReader(object):

    SUPPORTED_YUV_8BIT_TYPES = ['yuv420p',
                                'yuv422p',
                                'yuv444p',
                                ]

    SUPPORTED_YUV_10BIT_LE_TYPES = ['yuv420p10le',
                                    'yuv422p10le',
                                    'yuv444p10le',
                                    ]

    # ex: for yuv420p, the width and height of U/V is 0.5x, 0.5x of Y
    UV_WIDTH_HEIGHT_MULTIPLIERS_DICT = {'yuv420p': (0.5, 0.5),
                                        'yuv422p': (0.5, 1.0),
                                        'yuv444p': (1.0, 1.0),
                                        'yuv420p10le': (0.5, 0.5),
                                        'yuv422p10le': (0.5, 1.0),
                                        'yuv444p10le': (1.0, 1.0),
                                        }

    def __init__(self, filepath, width, height, yuv_type):

        self.filepath = filepath
        self.width = width
        self.height = height
        self.yuv_type = yuv_type

        self._asserts()

        self.file = open(self.filepath, 'rb')

    def close(self):
        self.file.close()

    # make YuvReader withable, e.g.:
    # with YuvReader(...) as yuv_reader:
    #     ...
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # make YuvReader iterable, e.g.:
    # for y, u, v in yuv_reader:
    #    ...
    def __iter__(self):
        return self
    def next(self):
        try:
            return self.next_y_u_v()
        except EOFError:
            raise StopIteration

    @property
    def num_bytes(self):
        self._assert_file_exist()
        return os.path.getsize(self.filepath)

    @property
    def num_frms(self):
        w_multiplier, h_multiplier = self._get_uv_width_height_multiplier()

        if self._is_10bitle():
            num_frms = float(self.num_bytes) / self.width / self.height \
                       / (1.0 + w_multiplier * h_multiplier * 2) / 2
        elif self._is_8bit():
            num_frms = float(self.num_bytes) / self.width / self.height \
                       / (1.0 + w_multiplier * h_multiplier * 2)
        else:
            assert False

        assert num_frms.is_integer(), \
            'Number of frames is not integer: {}'.format(num_frms)

        return int(num_frms)

    def _get_uv_width_height_multiplier(self):
        self._assert_yuv_type()
        return self.UV_WIDTH_HEIGHT_MULTIPLIERS_DICT[self.yuv_type]

    def _assert_yuv_type(self):
        assert (self.yuv_type in self.SUPPORTED_YUV_8BIT_TYPES
                or self.yuv_type in self.SUPPORTED_YUV_10BIT_LE_TYPES), \
            'Unsupported YUV type: {}'.format(self.yuv_type)

    def _assert_file_exist(self):
        assert os.path.exists(self.filepath), \
            "File does not exist: {}".format(self.filepath)

    def _asserts(self):

        # assert YUV type
        self._assert_yuv_type()

        # assert file exists
        self._assert_file_exist()

        # assert file size: if consists of integer number of frames
        num_frms = self.num_frms

    def _is_8bit(self):
        return self.yuv_type in self.SUPPORTED_YUV_8BIT_TYPES

    def _is_10bitle(self):
        return self.yuv_type in self.SUPPORTED_YUV_10BIT_LE_TYPES

    def next_y_u_v(self):

        y_width = self.width
        y_height = self.height
        uv_w_multiplier, uv_h_multiplier = self._get_uv_width_height_multiplier()
        uv_width = int(y_width * uv_w_multiplier)
        uv_height = int(y_height * uv_h_multiplier)

        if self._is_10bitle():
            pix_type = np.uint16
        elif self._is_8bit():
            pix_type = np.uint8
        else:
            assert False

        y = np.fromfile(self.file, pix_type, count=y_width*y_height)
        if y.size == 0:
            raise EOFError
        u = np.fromfile(self.file, pix_type, count=uv_width*uv_height)
        if u.size == 0:
            raise EOFError
        v = np.fromfile(self.file, pix_type, count=uv_width*uv_height)
        if v.size == 0:
            raise EOFError

        y = y.reshape(y_height, y_width)
        u = u.reshape(uv_height, uv_width)
        v = v.reshape(uv_height, uv_width)

        if self._is_10bitle():
            return y.astype(np.double) / 4.0, \
                   u.astype(np.double) / 4.0, \
                   v.astype(np.double) / 4.0
        elif self._is_8bit():
            return y.astype(np.double), \
                   u.astype(np.double), \
                   v.astype(np.double)
        else:
            assert False
