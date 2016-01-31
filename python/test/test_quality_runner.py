from asset import Asset
from quality_runner import VmafQualityRunner

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import unittest

class QualityRunnerTest(unittest.TestCase):
    def test_get_log_file_path(self):

        asset = Asset(dataset="test", ref_path="dir/refvideo.yuv",
                      dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})

        runner = VmafQualityRunner([asset], None)
        log_file_path = runner._get_log_file_path(asset)
        expected_log_file_path = "../workspace/log_file_dir/VMAF/" \
                          "test_refvideo_720x480_disvideo_720x480_frame2to2"
        self.assertEquals(log_file_path, expected_log_file_path)


if __name__ == '__main__':
    unittest.main()
