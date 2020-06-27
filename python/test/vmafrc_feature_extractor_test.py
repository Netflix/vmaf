import unittest

from vmaf.core.asset import Asset

from vmaf.config import VmafConfig

from test.testutil import set_default_576_324_videos_for_testing

from vmaf.core.vmafrc_feature_extractor import FloatMotionFeatureExtractor, IntegerMotionFeatureExtractor, FloatVifFeatureExtractor, FloatAdmFeatureExtractor, IntegerVifFeatureExtractor, IntegerPsnrFeatureExtractor


class FeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass

    def test_run_float_motion_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_score'], 3.8953518541666665, places=8)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_score'], 3.8953518541666665, places=8)

    def test_run_integer_motion_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_score'], 3.895345229166667, places=8)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_score'], 3.895345229166667, places=8)

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
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_score'], 0.9345148541666667, places=4)
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

    def test_run_float_adm_fextractor_akiyo_multiply(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = FloatAdmFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_score'], 1.116686, places=6)

    def test_run_float_adm_fextractor_akiyo_multiply_enhn_gain_limit_1(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = FloatAdmFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_enhn_gain_limit': 1.0}
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_score'], 0.9574308606115118, places=6)

    def test_run_float_adm_fextractor_akiyo_multiply_enhn_gain_limit_1d2(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = FloatAdmFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_enhn_gain_limit': 1.2}
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_score'], 1.116595, places=6)

    def test_run_float_vif_fextractor_akiyo_multiply(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = FloatVifFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 1.0522544319369052, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 1.0705609423182443, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 1.0731529493098957, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 1.0728060231246508, places=5)

    def test_run_float_vif_fextractor_akiyo_multiply_enhn_gain_limit_1(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = FloatVifFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_enhn_gain_limit': 1.0},
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 0.983699512450884, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 0.9974276726830457, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 0.9984692380091739, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 0.999146211879154, places=4)

    def test_run_float_vif_fextractor_akiyo_multiply_enhn_gain_limit_1d1(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = FloatVifFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_enhn_gain_limit': 1.1},
        )
        self.fextractor.run()
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 1.0298451531242514, places=3)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 1.046596975760772, places=3)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 1.0485607628500504, places=3)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 1.0491232394147363, places=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
