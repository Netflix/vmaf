import numpy as np

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class YuvWriter(object):

    SUPPORTED_YUV_8BIT_TYPES = ['yuv420p',
                                'yuv422p',
                                'yuv444p',
                                'gray',
                                ]

    SUPPORTED_YUV_10BIT_LE_TYPES = ['yuv420p10le',
                                    'yuv422p10le',
                                    'yuv444p10le',
                                    'gray10le',
                                    ]

    # ex: for yuv420p, the width and height of U/V is 0.5x, 0.5x of Y
    UV_WIDTH_HEIGHT_MULTIPLIERS_DICT = {'yuv420p': (0.5, 0.5),
                                        'yuv422p': (0.5, 1.0),
                                        'yuv444p': (1.0, 1.0),
                                        'gray': (0.0, 0.0),
                                        'yuv420p10le': (0.5, 0.5),
                                        'yuv422p10le': (0.5, 1.0),
                                        'yuv444p10le': (1.0, 1.0),
                                        'gray10le': (0.0, 0.0),
                                        }

    def __init__(self, filepath, width, height, yuv_type):

        self.filepath = filepath
        self.width = width
        self.height = height
        self.yuv_type = yuv_type

        self._asserts()

        self.file = open(self.filepath, 'wb')

    def close(self):
        self.file.close()

    # make YuvWriter withable, e.g.:
    # with YuvWriter(...) as yuv_writer:
    #     ...
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _is_8bit(self):
        return self.yuv_type in self.SUPPORTED_YUV_8BIT_TYPES

    def _is_10bitle(self):
        return self.yuv_type in self.SUPPORTED_YUV_10BIT_LE_TYPES

    def _get_uv_width_height_multiplier(self):
        self._assert_yuv_type()
        return self.UV_WIDTH_HEIGHT_MULTIPLIERS_DICT[self.yuv_type]

    def _assert_yuv_type(self):
        assert (self.yuv_type in self.SUPPORTED_YUV_8BIT_TYPES
                or self.yuv_type in self.SUPPORTED_YUV_10BIT_LE_TYPES), \
            'Unsupported YUV type: {}'.format(self.yuv_type)

    def _asserts(self):

        # assert YUV type
        self._assert_yuv_type()

    def next(self, y, u, v, format='uint'):

        assert format in ['uint', 'float2uint'], \
            "For now support two modes: \n" \
            "uint - directly map y, u, v values to the corresponding uint types; \n" \
            "float2uint - assume y, u, v are in [0, 1], do proper scaling and map to the corresponding uint types"

        y_width = self.width
        y_height = self.height
        uv_w_multiplier, uv_h_multiplier = self._get_uv_width_height_multiplier()
        uv_width = int(y_width * uv_w_multiplier)
        uv_height = int(y_height * uv_h_multiplier)

        assert (y_height, y_width) == y.shape

        if uv_width == 0 and uv_height == 0:
            assert u is None
            assert v is None
        elif uv_width > 0 and uv_height > 0:
            assert (uv_height, uv_width) == u.shape
            assert (uv_height, uv_width) == v.shape
        else:
            assert False, f'Unsupported uv_width and uv_height: {uv_width}, {uv_height}'

        if self._is_8bit():
            pix_type = np.uint8
        elif self._is_10bitle():
            pix_type = np.uint16
        else:
            assert False

        if format == 'uint':
            pass
        elif format == 'float2uint':
            if self._is_8bit():
                y = y.astype(np.double) * (2.0**8 - 1.0)
                u = u.astype(np.double) * (2.0**8 - 1.0) if u is not None else None
                v = v.astype(np.double) * (2.0**8 - 1.0) if v is not None else None
            elif self._is_10bitle():
                y = y.astype(np.double) * (2.0**10 - 1.0)
                u = u.astype(np.double) * (2.0**10 - 1.0) if u is not None else None
                v = v.astype(np.double) * (2.0**10 - 1.0) if v is not None else None
            else:
                assert False
        else:
            assert False

        self.file.write(y.astype(pix_type).tobytes())
        if u is not None:
            self.file.write(u.astype(pix_type).tobytes())
        if v is not None:
            self.file.write(v.astype(pix_type).tobytes())
