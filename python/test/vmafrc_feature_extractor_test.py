import unittest

from testutil import set_default_576_324_videos_for_testing

from vmaf.core.vmafrc_feature_extractor import FloatMotion2FeatureExtractor, IntegerMotion2FeatureExtractor, FloatVifFeatureExtractor, FloatAdmFeatureExtractor, IntegerVifFeatureExtractor, IntegerPsnrFeatureExtractor


class FeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass

    def test_run_float_motion2_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotion2FeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion2_feature_motion2_score'], 3.8953518541666665, places=8)
        self.assertAlmostEqual(results[1]['float_motion2_feature_motion2_score'], 3.8953518541666665, places=8)

    def test_run_integer_motion2_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerMotion2FeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_motion2_feature_motion2_score'], 3.895345229166667, places=8)
        self.assertAlmostEqual(results[1]['integer_motion2_feature_motion2_score'], 3.895345229166667, places=8)

    def test_run_float_vif_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 0.3634208125, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 0.7666474166666667, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 0.8628533333333334, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 0.9159719583333334, places=6)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_score'], 1.0, places=5)

    def test_run_integer_vif_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_score'], 0.3634233125, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale1_score'], 0.7666542916666667, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale2_score'], 0.862872125, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_score'], 0.9159956249999999, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale0_score'], 1.000002, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale1_score'], 1.0000023541666667, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale2_score'], 1.0000022916666667, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale3_score'], 1.0, places=5)

    def test_run_float_adm_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_score'], 0.9345877291666667, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_score'], 1.0, places=6)

    def test_run_integer_psnr_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_y_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_cb_score'], 38.4494410625, places=4)
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_cr_score'], 40.99191027083334, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_y_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_cb_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_cr_score'], 60.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
