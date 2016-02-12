__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
from feature_extractor import VmafFeatureExtractor

class FeatureExtractorTest(unittest.TestCase):

    def test_executor_id(self):
        fextractor = VmafFeatureExtractor()
        self.assertEquals(fextractor.executor_id, "VMAF_feature_V0.1")
