import unittest

import numpy as np

from test.testutil import set_default_576_324_videos_for_testing, \
    set_default_576_324_16bit_videos_for_testing, \
    set_default_576_324_12bit_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing_b
from vmaf.third_party.xiph.vmafexec_feature_extractor import PsnrhvsFeatureExtractor
from vmaf.tools.misc import MyTestCase


class FeatureExtractorTest(MyTestCase):

    def setUp(self):
        super().setUp()

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_run_psnrhvs_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = PsnrhvsFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_score'], 31.33044560416667, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_y_score'], 30.578766166666668, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cb_score'], 37.25849772916666, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cr_score'], 38.2002601875, places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_y_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cb_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cr_score'], float('inf'), places=4)

    def test_run_psnrhvs_fextractor_10bit_b(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing_b()
        self.fextractor = PsnrhvsFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_score'], 33.937514666666665, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_y_score'], 33.33393133333333, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cb_score'], 38.065045999999995, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cr_score'], 38.93428866666667, places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_y_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cb_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cr_score'], float('inf'), places=4)

    def test_run_psnrhvs_fextractor_12bit(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()
        self.fextractor = PsnrhvsFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_score'], 33.95532633333334, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_y_score'], 33.350897, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cb_score'], 38.088281333333335, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cr_score'], 38.97104833333333, places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_y_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cb_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cr_score'], float('inf'), places=4)

    def test_run_psnrhvs_fextractor_16bit(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()
        self.fextractor = PsnrhvsFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_score'], 33.95532633333334, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_y_score'], 33.350897, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cb_score'], 38.088281333333335, places=4)
        self.assertAlmostEqual(results[0]['PSNRHVS_feature_psnr_hvs_cr_score'], 38.97104833333333, places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_y_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cb_score'], float('inf'), places=4)
        self.assertAlmostEqual(results[1]['PSNRHVS_feature_psnr_hvs_cr_score'], float('inf'), places=4)
