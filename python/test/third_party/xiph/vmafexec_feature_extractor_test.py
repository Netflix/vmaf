import unittest

import numpy as np

from test.testutil import set_default_576_324_videos_for_testing
from vmaf.third_party.xiph.vmafexec_feature_extractor import PsnrhvsFeatureExtractor


class FeatureExtractorTest(unittest.TestCase):

    def setUp(self):
        self.verificationErrors = []
        self.maxDiff = None

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass
        self.assertEqual([], self.verificationErrors)

    def test_run_psnrhvs_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = PsnrhvsFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        try: self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_score'], 31.33044560416667, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_y_score'], 30.578766166666668, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cb_score'], 37.25849772916666, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cr_score'], 38.2002601875, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_score'], np.float('inf'), places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_y_score'], np.float('inf'), places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cb_score'], np.float('inf'), places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cr_score'], np.float('inf'), places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
