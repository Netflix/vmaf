__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
from feature_extractor import VmafFeatureExtractor
from asset import Asset

class FeatureExtractorTest(unittest.TestCase):

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={})
        fextractor = VmafFeatureExtractor([asset], None,
                                   log_file_dir="log_file_dir",
                                   fifo_mode=True)
        self.assertEquals(fextractor.executor_id, "VMAF_feature_V0.1")
