__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

import config
from tools.reader import YuvReader

class YuvReaderTest(unittest.TestCase):

    def test_yuv_reader(self):
        yuv_reader = YuvReader(
            filepath=config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv",
            width=576,
            height=324,
            yuv_type='yuv420p'
        )
        self.assertEquals(yuv_reader.num_bytes, 13436928)
        self.assertEquals(yuv_reader.num_frms, 48)