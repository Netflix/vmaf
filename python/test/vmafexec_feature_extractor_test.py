import unittest

import numpy as np

from vmaf.core.asset import Asset
from vmaf.config import VmafConfig
from vmaf.core.vmafexec_feature_extractor import FloatMotionFeatureExtractor, IntegerMotionFeatureExtractor, \
    FloatVifFeatureExtractor, FloatAdmFeatureExtractor, IntegerVifFeatureExtractor, IntegerPsnrFeatureExtractor, \
    IntegerAdmFeatureExtractor, CIEDE2000FeatureExtractor
from test.testutil import set_default_576_324_videos_for_testing, \
    set_default_576_324_12bit_videos_for_testing, \
    set_default_576_324_16bit_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing
from vmaf.tools.misc import MyTestCase


class FeatureExtractorTest(MyTestCase):

    def setUp(self):
        super().setUp()

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_run_float_motion_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_score'], 3.8953518541666665, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_score'], 3.8953518541666665, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_score']

    def test_run_float_motion_fextractor_with_debug(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_score'], 3.8953518541666665, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_score'], 3.8953518541666665, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion_score'], 4.0498253125, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion_score'], 4.0498253125, places=6)

    def test_run_float_motion_fextractor_forcing_zero(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_force_zero': True, 'debug': True},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_force_0_score'], 0.0, places=8)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_force_0_score'], 0.0, places=8)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion_force_0_score'], 0.0, places=8)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion_force_0_score'], 0.0, places=8)

        self.assertEqual(len(results[0]['float_motion_feature_motion2_force_0_scores']), 48)
        self.assertEqual(len(results[1]['float_motion_feature_motion2_force_0_scores']), 48)

    def test_run_integer_motion_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_score'], 3.895345229166667, places=8)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_score'], 3.895345229166667, places=8)
        with self.assertRaises(KeyError):
            s = results[0]['integer_motion_feature_motion_score']

    def test_run_integer_motion_fextractor_with_debug(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_score'], 3.895345229166667, places=8)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_score'], 3.895345229166667, places=8)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion_score'], 4.0498181041666665, places=8)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion_score'], 4.0498181041666665, places=8)

    def test_run_integer_motion_fextractor_forcing_zero(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_force_zero': True, 'debug': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_force_0_score'], 0.0, places=8)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_force_0_score'], 0.0, places=8)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion_force_0_score'], 0.0, places=8)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion_force_0_score'], 0.0, places=8)

    def test_run_integer_motion_fextractor_12bit(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()
        self.fextractor = IntegerMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_score'], 2.8104533333333332, places=8)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_score'], 2.8104533333333332, places=8)

    def test_run_float_vif_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 0.3634208125, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 0.7666474166666667, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 0.8628533333333334, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 0.9159719583333334, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_score'], 1.0, places=5)
        with self.assertRaises(KeyError):
            s = results[0]['float_VIF_feature_vif_num_score']

    def test_run_float_vif_fextractor_with_debug(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 0.3634208125, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 0.7666474166666667, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 0.8628533333333334, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 0.9159719583333334, places=6)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_score'], 1.0, places=5)

        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_score'], 0.44609339583333335, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_score'], 712650.1518554376, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_score'], 1597314.4783325624, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_scale0_score'], 468101.7565104167, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_scale0_score'], 1287822.3411458333, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_scale1_score'], 184971.52506510416, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_scale1_score'], 241255.05696614584, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_scale2_score'], 47588.75968416667, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_scale2_score'], 55149.814208979165, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_scale3_score'], 11988.110595750002, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_scale3_score'], 13087.266011562499, places=0)

    def test_run_integer_vif_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_score'], 0.3636620625, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale1_score'], 0.7674953125, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale2_score'], 0.8631078125, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_score'], 0.9157200833333333, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale0_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale3_score'], 1.0, places=5)
        with self.assertRaises(KeyError):
            s = results[0]['integer_VIF_feature_vif_num_score']

    def test_run_integer_vif_fextractor_with_debug(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_score'], 0.3636620625, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale1_score'], 0.7674953125, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale2_score'], 0.8631078125, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_score'], 0.9157200833333333, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale0_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale3_score'], 1.0, places=5)

        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_score'], 0.44642331250000006, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_num_score'], 713111.410502125, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_den_score'], 1597165.5464884583, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_num_scale0_score'], 468372.4192708333, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_den_scale0_score'], 1287711.328125, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_num_scale1_score'], 185125.40266927084, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_den_scale1_score'], 241188.595703125, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_num_scale2_score'], 47606.0171713125, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_den_scale2_score'], 55153.50821933334, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_num_scale3_score'], 12007.571390770832, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_den_scale3_score'], 13112.1144409375, places=4)

    def test_run_integer_vif_fextractor_12bit(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()
        self.fextractor = IntegerVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_score'], 0.4330893333333334, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale1_score'], 0.830613, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale2_score'], 0.9072123333333333, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_score'], 0.945896, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale0_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale3_score'], 1.0, places=5)

    def test_run_integer_vif_fextractor_debug1_yuv422p10le(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing()
        self.fextractor = IntegerVifFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_scores'][-1], 0.416638, places=6)

    def test_run_integer_vif_fextractor_debug2_160x90(self):
        ref_path = VmafConfig.test_resource_path("yuv", "ref_test_0_1_src01_hrc00_576x324_576x324_vs_src01_hrc01_576x324_576x324_q_160x90.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "dis_test_0_1_src01_hrc00_576x324_576x324_vs_src01_hrc01_576x324_576x324_q_160x90.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 160, 'height': 90})
        self.fextractor = IntegerVifFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_scores'][31], 0.982747, places=6)

    def test_run_integer_vif_fextractor_debug3_yuv420p10le(self):
        ref_path = VmafConfig.test_resource_path("yuv", "sparks_ref_480x270.yuv42010le.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "sparks_dis_480x270.yuv42010le.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 480, 'height': 270,
                                  'yuv_type': 'yuv420p10le'})
        self.fextractor = IntegerVifFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_scores'][0], 0.933186, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_scores'][2], 0.999348, places=6)

    def test_run_float_adm_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_score'], 0.9649663541666667, places=4)
        with self.assertRaises(KeyError):
            s = results[0]['float_ADM_feature_adm_num_score']

    def test_run_float_adm_fextractor_with_default_adm_csf_mode(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 0}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_score'], 0.9649663541666667, places=4)

    def test_run_float_adm_fextractor_with_debug(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': True},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_score'], 0.9649663541666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_score'], 0.9345148541666667, places=4)

        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_num_score'], 371.80645372916666, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_den_score'], 397.83379106250004, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_num_scale0_score'], 45.526146958333335, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_den_scale0_score'], 50.14385129166667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_num_scale1_score'], 66.574236, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_den_scale1_score'], 74.47438383333333, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_num_scale2_score'], 105.55483329166668, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_den_scale2_score'], 113.49725864583333, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_num_scale3_score'], 154.15123754166666, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_den_scale3_score'], 159.7182974375, places=4)

    def test_run_integer_psnr_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_y_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_cb_score'], 38.4494410625, places=4)
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_cr_score'], 40.99191027083334, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_y_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_cb_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_cr_score'], 60.0, places=4)

    def test_run_integer_psnr_fextractor_12bit(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()
        self.fextractor = IntegerPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_y_score'], 32.577818, places=4)
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_cb_score'], 39.044961, places=4)
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_cr_score'], 41.286965333333335, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_y_score'], 84.0, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_cb_score'], 84.0, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_cr_score'], 84.0, places=4)

    def test_run_integer_psnr_fextractor_16bit(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()
        self.fextractor = IntegerPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_y_score'], 32.579806000000005, places=4)
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_cb_score'], 39.04694966666667, places=4)
        self.assertAlmostEqual(results[0]['integer_PSNR_feature_psnr_cr_score'], 41.288954, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_y_score'], 108.0, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_cb_score'], 108.0, places=4)
        self.assertAlmostEqual(results[1]['integer_PSNR_feature_psnr_cr_score'], 108.0, places=4)

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
        self.fextractor.run(parallelize=True)
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
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_egl_1_score'], 0.9574308606115118, places=6) #

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
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_egl_1.2_score'], 1.116595, places=6)

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
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 1.0522544319369052, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 1.0705609423182443, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 1.0731529493098957, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 1.0728060231246508, places=4)

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
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_egl_1_score'], 0.983699512450884, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_egl_1_score'], 0.9974276726830457, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_egl_1_score'], 0.9984692380091739, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_egl_1_score'], 0.999146211879154, places=4)

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
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_egl_1.1_score'], 1.0298451531242514, places=3)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_egl_1.1_score'], 1.046596975760772, places=3)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_egl_1.1_score'], 1.0485607628500504, places=3)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_egl_1.1_score'], 1.0491232394147363, places=3)

    def test_run_integer_vif_fextractor_akiyo_multiply(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = IntegerVifFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_score'], 1.052403, places=5)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale1_score'], 1.070149, places=5)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale2_score'], 1.072518, places=5)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_score'], 1.072512, places=5)

    def test_run_integer_vif_fextractor_akiyo_multiply_enhn_gain_limit_1(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = IntegerVifFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_enhn_gain_limit': 1.0},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_egl_1_score'], 0.983699512450884, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale1_egl_1_score'], 0.9974276726830457, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale2_egl_1_score'], 0.9984692380091739, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_egl_1_score'], 0.999146211879154, places=4)

    def test_run_integer_vif_fextractor_akiyo_multiply_enhn_gain_limit_1d1(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = IntegerVifFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_enhn_gain_limit': 1.1},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_egl_1.1_score'], 1.0298451531242514, places=3)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale1_egl_1.1_score'], 1.046596975760772, places=3)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale2_egl_1.1_score'], 1.0485607628500504, places=3)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_egl_1.1_score'], 1.0491232394147363, places=3)

    def test_run_integer_adm_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_score'], 0.9345057916666667, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_score'], 1.000002, places=6)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_score'], 0.9649663541666667, places=4)
        with self.assertRaises(KeyError):
            s = results[0]['integer_adm_num_score']

    def test_run_integer_adm_fextractor_with_debug(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': True},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_score'], 0.9345057916666667, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_score'], 1.000002, places=6)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_score'], 0.9649663541666667, places=4)

        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_num_score'], 371.8243668541666, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_den_score'], 397.8567857291667, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_num_scale0_score'], 45.526718708333334, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_den_scale0_score'], 50.14397566666668, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_num_scale1_score'], 66.57503025, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_den_scale1_score'], 74.47584229166667, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_num_scale2_score'], 105.5588711875, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_den_scale2_score'], 113.50324393750002, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_num_scale3_score'], 154.1637464375, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_den_scale3_score'], 159.733723875, places=4)

    def test_run_integer_adm_fextractor_12bit(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_score'], 0.9517706666666667, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_score'], 1.000002, places=6)

    def test_run_integer_adm_fextractor_akiyo_multiply(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_score'], 1.1167, places=5)  # float 1.116686

    def test_run_integer_adm_fextractor_akiyo_multiply_enhn_gain_limit_1(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_enhn_gain_limit': 1.0}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_egl_1_score'], 0.957433, places=6)  # float 0.9574308606115118

    def test_run_integer_adm_fextractor_akiyo_multiply_enhn_gain_limit_1d2(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_enhn_gain_limit': 1.2}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_egl_1.2_score'], 1.116609, places=5)  # float 1.116595

    def test_run_ciede2000_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CIEDE2000FeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['CIEDE2000_feature_ciede2000_score'], 33.10755745833333, places=4)
        self.assertAlmostEqual(results[1]['CIEDE2000_feature_ciede2000_score'], float('inf'), places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
