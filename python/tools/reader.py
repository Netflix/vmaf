__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os

class YuvReader(object):

    SUPPORTED_YUV_TYPES = ['yuv420p', 'yuv422p', 'yuv444p']

    # ex: for yuv420p, the width and height of U/V is 0.5x, 0.5x of Y
    UV_WIDTH_HEIGHT_MULTIPLIERS_DICT = {'yuv420p': (0.5, 0.5),
                                        'yuv422p': (0.5, 1.0),
                                        'yuv444p': (1.0, 1.0)}

    def __init__(self, filepath, width, height, yuv_type):

        self.filepath = filepath
        self.width = width
        self.height = height
        self.yuv_type = yuv_type

        self._asserts()

        self.file = open(self.filepath, 'rb')

    @property
    def num_bytes(self):
        self._assert_file_exist()
        return os.path.getsize(self.filepath)

    @property
    def num_frms(self):
        w_multiplier, h_multiplier = self._get_uv_width_height_multiplier()
        num_frms = float(self.num_bytes) / self.width / self.height \
                   / (1.0 + w_multiplier * h_multiplier * 2)

        assert num_frms.is_integer(), \
            'Number of frames is not integer: {}'.format(num_frms)

        return int(num_frms)

    def _get_uv_width_height_multiplier(self):
        self._assert_yuv_type()
        return self.UV_WIDTH_HEIGHT_MULTIPLIERS_DICT[self.yuv_type]

    def _assert_yuv_type(self):
        assert self.yuv_type in self.SUPPORTED_YUV_TYPES, \
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

    def next_y_u_v(self):
        pass



