import unittest


from vmaf.core.asset import Asset
from vmaf.config import VmafConfig
from vmaf.core.vmafexec_feature_extractor import FloatMotionFeatureExtractor, IntegerMotionFeatureExtractor, \
    FloatVifFeatureExtractor, FloatAdmFeatureExtractor, IntegerVifFeatureExtractor, IntegerPsnrFeatureExtractor, \
    IntegerAdmFeatureExtractor, CIEDE2000FeatureExtractor
from test.testutil import set_default_576_324_videos_for_testing, \
    set_default_576_324_videos_for_testing_5frames, \
    set_default_576_324_12bit_videos_for_testing, \
    set_default_576_324_16bit_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing
from vmaf.tools.misc import MyTestCase


class FeatureExtractorTest(MyTestCase):

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
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_scores'][0], 4.57945, places=6)
        assert len(results[0]['float_motion_feature_motion2_scores']) == \
               len(results[0]['float_motion_feature_motion3_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_scores'][i],
                             results[0]['float_motion_feature_motion3_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_score'], 3.894366229166667, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_score'], 3.894366229166667, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_score']

    def test_run_float_motion_fextractor_motion_fps_weight_2d5(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False, 'motion_fps_weight': 2.5}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mfw_2.5_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mfw_2.5_scores'][0], 11.448624, places=6)
        assert len(results[0]['float_motion_feature_motion2_mfw_2.5_scores']) == \
               len(results[0]['float_motion_feature_motion3_mfw_2.5_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_mfw_2.5_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_mfw_2.5_scores'][i],
                             results[0]['float_motion_feature_motion3_mfw_2.5_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mfw_2.5_score'], 9.735915666666665, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_mfw_2.5_score'], 9.735915666666665, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_mfw_2.5_score']

    def test_run_float_motion_fextractor_add_scale1(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False, 'motion_add_scale1': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mdc_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mdc_scores'][0], 8.758978, places=6)
        assert len(results[0]['float_motion_feature_motion2_mdc_scores']) == \
               len(results[0]['float_motion_feature_motion3_mdc_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_mdc_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_mdc_scores'][i],
                             results[0]['float_motion_feature_motion3_mdc_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mdc_score'], 7.472933979166666, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_mdc_score'], 7.472933979166666, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_mdc_score']

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
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_scores'][0], 4.57945, places=6)
        assert len(results[0]['float_motion_feature_motion2_scores']) == \
               len(results[0]['float_motion_feature_motion3_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_scores'][i],
                             results[0]['float_motion_feature_motion3_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_score'], 3.894366229166667, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_score'], 3.894366229166667, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion_score'], 4.0488208125, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion_score'], 4.0488208125, places=6)

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
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_force_0_score'], 0.0, places=8)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion3_force_0_score'], 0.0, places=8)

        self.assertEqual(len(results[0]['float_motion_feature_motion2_force_0_scores']), 48)
        self.assertEqual(len(results[1]['float_motion_feature_motion2_force_0_scores']), 48)
        self.assertEqual(len(results[0]['float_motion_feature_motion_force_0_scores']), 48)
        self.assertEqual(len(results[1]['float_motion_feature_motion_force_0_scores']), 48)
        self.assertEqual(len(results[0]['float_motion_feature_motion3_force_0_scores']), 48)
        self.assertEqual(len(results[1]['float_motion_feature_motion3_force_0_scores']), 48)

    def test_run_float_motion_fextractor_with_blending(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_blend_factor': 0.5, 'motion_blend_offset': 3.0}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mbf_0.5_mbo_3_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mbf_0.5_mbo_3_scores'][0], 3.789725, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mbf_0.5_mbo_3_scores'][1], 3.607168, places=5)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mbf_0.5_mbo_3_scores'][2], 3.535807, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mbf_0.5_mbo_3_scores'][3], 3.412849, places=5)
        assert len(results[0]['float_motion_feature_motion2_mbf_0.5_mbo_3_scores']) == \
               len(results[0]['float_motion_feature_motion3_mbf_0.5_mbo_3_scores'])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mbf_0.5_mbo_3_score'], 3.894366229166667, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mbf_0.5_mbo_3_score'], 3.494885791666667, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_mbf_0.5_mbo_3_score'], 3.894366229166667, places=6)

    def test_run_float_motion_fextractor_no_filter(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False, 'motion_filter_size': 1}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mfs_1_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mfs_1_scores'][0], 10.197686, places=6)
        assert len(results[0]['float_motion_feature_motion2_mfs_1_scores']) == \
               len(results[0]['float_motion_feature_motion3_mfs_1_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_mfs_1_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_mfs_1_scores'][i],
                             results[0]['float_motion_feature_motion3_mfs_1_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mfs_1_score'], 8.1035535, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_mfs_1_score'], 8.1035535, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_mfs_1_score']

    def test_run_float_motion_fextractor_add_uv(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False, 'motion_add_uv': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mau_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mau_scores'][0], 5.820648, places=6)
        assert len(results[0]['float_motion_feature_motion2_mau_scores']) == \
               len(results[0]['float_motion_feature_motion3_mau_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_mau_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_mau_scores'][i],
                             results[0]['float_motion_feature_motion3_mau_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mau_score'], 5.0747014791666665, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_mau_score'], 5.0747014791666665, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_mau_score']

    def test_run_float_motion_fextractor_no_filter_add_uv(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False, 'motion_filter_size': 1, 'motion_add_uv': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mau_mfs_1_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mau_mfs_1_scores'][0], 13.203752, places=6)
        assert len(results[0]['float_motion_feature_motion2_mau_mfs_1_scores']) == \
               len(results[0]['float_motion_feature_motion3_mau_mfs_1_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_mau_mfs_1_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_mau_mfs_1_scores'][i],
                             results[0]['float_motion_feature_motion3_mau_mfs_1_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mau_mfs_1_score'], 10.760497729166666, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_mau_mfs_1_score'], 10.760497729166666, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_mau_mfs_1_score']

    def test_run_float_motion_fextractor_yuv42210ple_no_filter_add_uv(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False, 'motion_filter_size': 1, 'motion_add_uv': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mau_mfs_1_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mau_mfs_1_scores'][0], 13.160226, places=6)
        assert len(results[0]['float_motion_feature_motion2_mau_mfs_1_scores']) == \
               len(results[0]['float_motion_feature_motion3_mau_mfs_1_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_mau_mfs_1_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_mau_mfs_1_scores'][i],
                             results[0]['float_motion_feature_motion3_mau_mfs_1_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mau_mfs_1_score'], 10.746016395833331, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_mau_mfs_1_score'], 10.746016395833331, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_mau_mfs_1_score']

    def test_run_float_motion_fextractor_three_tap_gaussian(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False, 'motion_filter_size': 3}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mfs_3_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mfs_3_scores'][0], 6.702473, places=5)
        assert len(results[0]['float_motion_feature_motion2_mfs_3_scores']) == \
               len(results[0]['float_motion_feature_motion3_mfs_3_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_mfs_3_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_mfs_3_scores'][i],
                             results[0]['float_motion_feature_motion3_mfs_3_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mfs_3_score'], 5.481617229166667, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_mfs_3_score'], 5.481617229166667, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_mfs_3_score']

    def test_run_float_motion_fextractor_three_tap_gaussian_add_uv(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False, 'motion_filter_size': 3, 'motion_add_uv': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mau_mfs_3_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_motion_feature_motion3_mau_mfs_3_scores'][0], 8.594293, places=5)
        assert len(results[0]['float_motion_feature_motion2_mau_mfs_3_scores']) == \
               len(results[0]['float_motion_feature_motion3_mau_mfs_3_scores'])
        for i in range(1, len(results[0]['float_motion_feature_motion2_mau_mfs_3_scores'])):
            self.assertEqual(results[0]['float_motion_feature_motion2_mau_mfs_3_scores'][i],
                             results[0]['float_motion_feature_motion3_mau_mfs_3_scores'][i])
        self.assertAlmostEqual(results[0]['float_motion_feature_motion2_mau_mfs_3_score'], 7.235332708333332, places=6)
        self.assertAlmostEqual(results[1]['float_motion_feature_motion2_mau_mfs_3_score'], 7.235332708333332, places=6)
        with self.assertRaises(KeyError):
            s = results[0]['float_motion_feature_motion_mau_mfs_2_score']

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
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_scores'][0], 4.579442, places=6)
        assert len(results[0]['integer_motion_feature_motion2_scores']) == \
               len(results[0]['integer_motion_feature_motion3_scores'])
        for i in range(1, len(results[0]['integer_motion_feature_motion2_scores'])):
            self.assertEqual(results[0]['integer_motion_feature_motion2_scores'][i],
                             results[0]['integer_motion_feature_motion3_scores'][i])
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_score'], 3.8943597291666667, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_score'], 3.8943597291666667, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_score'], 3.9897647708333337, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion3_score'], 3.9897647708333337, places=5)
        with self.assertRaises(KeyError):
            s = results[0]['integer_motion_feature_motion_score']

    def test_run_integer_motion_fextractor_motion_fps_weight(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'debug': False, 'motion_fps_weight': 2.5},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_mfw_2.5_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_mfw_2.5_scores'][0], 11.448605, places=6)
        assert len(results[0]['integer_motion_feature_motion2_mfw_2.5_scores']) == \
               len(results[0]['integer_motion_feature_motion3_mfw_2.5_scores'])
        for i in range(1, len(results[0]['integer_motion_feature_motion2_mfw_2.5_scores'])):
            self.assertEqual(results[0]['integer_motion_feature_motion2_mfw_2.5_scores'][i],
                             results[0]['integer_motion_feature_motion3_mfw_2.5_scores'][i])
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_mfw_2.5_score'], 9.735899000000002, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_mfw_2.5_score'], 9.735899000000002, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_mfw_2.5_score'], 9.974411604166667, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion3_mfw_2.5_score'], 9.974411604166667, places=5)
        with self.assertRaises(KeyError):
            s = results[0]['integer_motion_feature_motion_mfw_2.5_score']

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
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_scores'][0], 4.579442, places=6)
        assert len(results[0]['integer_motion_feature_motion2_scores']) == \
               len(results[0]['integer_motion_feature_motion3_scores'])
        for i in range(1, len(results[0]['integer_motion_feature_motion2_scores'])):
            self.assertEqual(results[0]['integer_motion_feature_motion2_scores'][i],
                             results[0]['integer_motion_feature_motion3_scores'][i])
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_score'], 3.8943597291666667, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_score'], 3.8943597291666667, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion_score'], 4.048813375, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion_score'], 4.048813375, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_score'], 3.9897647708333337, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion3_score'], 3.9897647708333337, places=5)

    def test_run_integer_motion_fextractor_with_blend(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerMotionFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_blend_factor': 0.5, 'motion_blend_offset': 3.0},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        print(results[0])
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_mbf_0.5_mbo_3_scores'][0], 0.0, places=6)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_mbf_0.5_mbo_3_scores'][0], 3.789721, places=6)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_mbf_0.5_mbo_3_scores'][1], 3.607162, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_mbf_0.5_mbo_3_scores'][2], 3.535804, places=4)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_mbf_0.5_mbo_3_scores'][3], 3.412845, places=5)
        assert len(results[0]['integer_motion_feature_motion2_mbf_0.5_mbo_3_scores']) == \
               len(results[0]['integer_motion_feature_motion3_mbf_0.5_mbo_3_scores'])
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_mbf_0.5_mbo_3_score'], 3.8943597291666667, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_mbf_0.5_mbo_3_score'], 3.8943597291666667, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_mbf_0.5_mbo_3_score'], 3.494882270833333, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion3_mbf_0.5_mbo_3_score'], 3.494882270833333, places=5)

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
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_force_0_score'], 0.0, places=8)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion3_force_0_score'], 0.0, places=8)
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
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_scores'][0], 0.0, places=8)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_scores'][1], 4.214324, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_scores'][2], 4.214324, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion2_score'], 2.8095493333333335, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_score'], 4.336030000000001, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_scores'][0], 4.579442, places=8)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_scores'][1], 4.214324, places=5)
        self.assertAlmostEqual(results[0]['integer_motion_feature_motion3_scores'][2], 4.214324, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion2_score'], 2.8095493333333335, places=5)
        self.assertAlmostEqual(results[1]['integer_motion_feature_motion3_score'], 4.336030000000001, places=5)

    def test_run_float_vif_fextractor(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 0.3636628333333333, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 0.7674898958333333, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 0.8630880833333334, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 0.9156987499999999, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_score'], 1.0, places=5)
        with self.assertRaises(KeyError):
            s = results[0]['float_VIF_feature_vif_num_score']

    def test_run_float_vif_fextractor_prescale_nearest_0d5(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 0.5},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_0.5_score'], 0.44294898819774114, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_0.5_score'], 0.9041909417379965, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_0.5_score'], 0.9531142394238457, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_0.5_score'], 0.9728893154941552, places=4)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_0.5_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_0.5_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_0.5_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_0.5_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_ks1o2d25(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_kernelscale": 0.444444},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ks_0.444444_score'], 0.37937579999999993, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ks_0.444444_score'], 0.662094, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ks_0.444444_score'], 0.7968873999999999, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ks_0.444444_score'], 0.9048748, places=4)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ks_0.444444_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ks_0.444444_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ks_0.444444_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ks_0.444444_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_nearest_2(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 2},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_2_score'], 0.37738296687637224, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_2_score'], 0.6416248740008155, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_2_score'], 0.7537032961151853, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_2_score'], 0.8334312405504697, places=4)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_2_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_nearest_0d3333(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 0.3333, "vif_prescale_method": "nearest"},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_0.3333_score'], 0.459471, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_0.3333_score'], 0.9075554, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_0.3333_score'], 0.9539954, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_0.3333_score'], 0.9733826, places=4)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_0.3333_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_0.3333_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_0.3333_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_0.3333_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_bicubic_0d5(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 0.5, "vif_prescale_method": "bicubic"},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_0.5_pm_bicubic_score'], 0.480674, places=4)  # python cv2: 0.48096469552121873
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_0.5_pm_bicubic_score'], 0.926664, places=4)  # python cv2: 0.926819738377272
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_0.5_pm_bicubic_score'], 0.964818, places=4)  # python cv2: 0.9648579489126992
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_0.5_pm_bicubic_score'], 0.979879, places=4)  # python cv2: 0.9799167928639173
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_0.5_pm_bicubic_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_0.5_pm_bicubic_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_0.5_pm_bicubic_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_0.5_pm_bicubic_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_bicubic_0d3333(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 0.3333, "vif_prescale_method": "bicubic"},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_0.3333_pm_bicubic_score'], 0.461591, places=4)  # python cv2: 0.4618999314294322
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_0.3333_pm_bicubic_score'], 0.906136, places=4)  # python cv2: 0.9064007316865517
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_0.3333_pm_bicubic_score'], 0.953404, places=4)  # python cv2: 0.9535245702897566
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_0.3333_pm_bicubic_score'], 0.973297, places=4)  # python cv2: 0.9734059800378327
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_0.3333_pm_bicubic_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_0.3333_pm_bicubic_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_0.3333_pm_bicubic_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_0.3333_pm_bicubic_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_bicubic_2(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 2, "vif_prescale_method": "bicubic"},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_2_pm_bicubic_score'], 0.401549, places=4)  # python cv2: 0.4014591349964812
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_2_pm_bicubic_score'], 0.628202, places=4)  # python cv2: 0.6282133567657787
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_2_pm_bicubic_score'], 0.745137, places=4)  # python cv2: 0.7451741601050006
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_2_pm_bicubic_score'], 0.829127, places=4)  # python cv2: 0.8291485778631007
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_2_pm_bicubic_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_2_pm_bicubic_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_2_pm_bicubic_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_2_pm_bicubic_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_lanczos_0d5(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 0.5, "vif_prescale_method": "lanczos4"},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_0.5_pm_lanczos4_score'], 0.462732, places=4)  # python cv2: 0.462981
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_0.5_pm_lanczos4_score'], 0.925294, places=4)  # python cv2: 0.925616
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_0.5_pm_lanczos4_score'], 0.964466, places=4)  # python cv2: 0.964556
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_0.5_pm_lanczos4_score'], 0.979701, places=4)  # python cv2: 0.979779
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_0.5_pm_lanczos4_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_0.5_pm_lanczos4_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_0.5_pm_lanczos4_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_0.5_pm_lanczos4_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_lanczos_0d3333(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 0.3333, "vif_prescale_method": "lanczos4"},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_0.3333_pm_lanczos4_score'], 0.461591, places=4)  # python cv2: 0.4617452678580717
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_0.3333_pm_lanczos4_score'], 0.906136, places=4)  # python cv2: 0.9062409173997763
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_0.3333_pm_lanczos4_score'], 0.953404, places=4)  # python cv2: 0.9534330173098132
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_0.3333_pm_lanczos4_score'], 0.973297, places=4)  # python cv2: 0.9733639369907998
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_0.3333_pm_lanczos4_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_0.3333_pm_lanczos4_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_0.3333_pm_lanczos4_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_0.3333_pm_lanczos4_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_lanczos_2(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 2, "vif_prescale_method": "lanczos4"},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_2_pm_lanczos4_score'], 0.389396, places=4)  # python cv2: 0.38936530967436755
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_2_pm_lanczos4_score'], 0.631164, places=4)  # python cv2: 0.6311712541859923
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_2_pm_lanczos4_score'], 0.748926, places=4)  # python cv2: 0.7489309041803393
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_2_pm_lanczos4_score'], 0.831347, places=4)  # python cv2: 0.8313460328738558
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_2_pm_lanczos4_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_2_pm_lanczos4_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_2_pm_lanczos4_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_2_pm_lanczos4_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_bilinear_2(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 2, "vif_prescale_method": "bilinear"},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_2_pm_bilinear_score'], 0.464984, places=4)  # python cv2: 0.46563038832805576
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_2_pm_bilinear_score'], 0.662918, places=4)  # python cv2: 0.6626177042209648
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_2_pm_bilinear_score'], 0.764193, places=4)  # python cv2: 0.7640623214864835
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_2_pm_bilinear_score'], 0.838125, places=4)  # python cv2: 0.8381248418471238
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_2_pm_bilinear_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_2_pm_bilinear_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_2_pm_bilinear_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_2_pm_bilinear_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_prescale_bilinear_0d5(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"vif_prescale": 0.5, "vif_prescale_method": "bilinear"},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_ps_0.5_pm_bilinear_score'], 0.555946, places=4)  # python cv2: 0.5559478961661829
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ps_0.5_pm_bilinear_score'], 0.929167, places=4)  # python cv2: 0.9291678487990115
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ps_0.5_pm_bilinear_score'], 0.965469, places=4)  # python cv2: 0.9654703777029944
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ps_0.5_pm_bilinear_score'], 0.980117, places=4)  # python cv2: 0.9801164123421076
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_ps_0.5_pm_bilinear_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ps_0.5_pm_bilinear_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ps_0.5_pm_bilinear_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ps_0.5_pm_bilinear_score'], 1.0, places=5)

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
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 0.3636628333333333, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 0.7674898958333333, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 0.8630880833333334, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 0.9156987499999999, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale0_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_score'], 1.0, places=5)

        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_score'], 0.44641939583333334, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_score'], 713112.2182617291, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_score'], 1597172.416483604, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_scale0_score'], 468375.916015625, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_scale0_score'], 1287718.2395833333, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_scale1_score'], 185123.1435546875, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_scale1_score'], 241187.58854166666, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_scale2_score'], 47605.012532562505, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_scale2_score'], 55154.17488602083, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_num_scale3_score'], 12007.192342145834, places=0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_den_scale3_score'], 13112.004638666665, places=0)

    def test_run_float_vif_fextractor_with_vif_skip_scale0(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_skip_scale0': True},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertEqual(results[0]['float_VIF_feature_vif_scale0_ssclz_score'], 0.0)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_ssclz_score'], 0.7674898958333333, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_ssclz_score'], 0.8630880833333334, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_ssclz_score'], 0.9156987499999999, places=5)
        self.assertEqual(results[1]['float_VIF_feature_vif_scale0_ssclz_score'], 0.0)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_ssclz_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_ssclz_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_ssclz_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_with_vif_sigma_nsq(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_sigma_nsq': 1.5},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_snsq_1.5_score'], 0.34616335416666666, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_snsq_1.5_score'], 0.7433470833333334, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_snsq_1.5_score'], 0.8450055416666666, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_snsq_1.5_score'], 0.9031788124999999, places=6)
        self.assertEqual(results[1]['float_VIF_feature_vif_scale0_snsq_1.5_score'], 1.0)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_snsq_1.5_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_snsq_1.5_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_snsq_1.5_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_with_vif_scale1_min_val(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_scale1_min_val': 0.987654},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_s1miv_0.987654_score'], 0.3636628333333333, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_s1miv_0.987654_score'], 0.987654, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_s1miv_0.987654_score'], 0.8630880833333334, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_s1miv_0.987654_score'], 0.9156987499999999, places=5)
        self.assertEqual(results[1]['float_VIF_feature_vif_scale0_s1miv_0.987654_score'], 1.0)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_s1miv_0.987654_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_s1miv_0.987654_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_s1miv_0.987654_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_with_vif_scale2_min_val(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_scale2_min_val': 0.987654},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        print(results[0])
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_s2miv_0.987654_score'], 0.3636628333333333, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_s2miv_0.987654_score'], 0.7674898958333333, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_s2miv_0.987654_score'], 0.987654, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_s2miv_0.987654_score'], 0.9156987499999999, places=5)
        self.assertEqual(results[1]['float_VIF_feature_vif_scale0_s2miv_0.987654_score'], 1.0)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_s2miv_0.987654_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_s2miv_0.987654_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_s2miv_0.987654_score'], 1.0, places=5)

    def test_run_float_vif_fextractor_with_vif_scale3_min_val(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_scale3_min_val': 0.987654},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        print(results[0])
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_s3miv_0.987654_score'], 0.3636628333333333, places=6)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_s3miv_0.987654_score'], 0.7674898958333333, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_s3miv_0.987654_score'], 0.8630880833333334, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_s3miv_0.987654_score'], 0.987654, places=6)
        self.assertEqual(results[1]['float_VIF_feature_vif_scale0_s3miv_0.987654_score'], 1.0)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale1_s3miv_0.987654_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale2_s3miv_0.987654_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['float_VIF_feature_vif_scale3_s3miv_0.987654_score'], 1.0, places=5)

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

    def test_run_integer_vif_fextractor_with_vif_skip_scale0(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerVifFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_skip_scale0': True},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertEqual(results[0]['integer_VIF_feature_vif_scale0_ssclz_score'], 0.0)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale1_ssclz_score'], 0.7674953125, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale2_ssclz_score'], 0.8631078125, places=6)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_ssclz_score'], 0.9157200833333333, places=6)
        self.assertEqual(results[1]['integer_VIF_feature_vif_scale0_ssclz_score'], 0.0)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale1_ssclz_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale2_ssclz_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['integer_VIF_feature_vif_scale3_ssclz_score'], 1.0, places=5)

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
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_score'], 0.9539779375, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_score'], 0.9649663541666667, places=4)
        with self.assertRaises(KeyError):
            s = results[0]['float_ADM_feature_adm_num_score']

    def test_run_float_adm_fextractor_apply_hm_adm3(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_adm3_apply_hm': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_aah_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_aah_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_aah_score'], 0.0516021875, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_aah_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_aah_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_aah_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_aah_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_aah_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_aah_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_aah_score'], 0.9649663541666667, places=4)
        with self.assertRaises(KeyError):
            s = results[0]['float_ADM_feature_adm_num_score']

    def test_run_float_adm_fextractor_skip_aim_scale_0(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_skip_aim_scale': 0}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_sasc_0_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_sasc_0_score'], 0.023317166666666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_sasc_0_score'], 0.9555988750000001, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_sasc_0_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_sasc_0_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_sasc_0_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_sasc_0_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_sasc_0_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_sasc_0_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_sasc_0_score'], 0.9649663541666667, places=4)
        with self.assertRaises(KeyError):
            s = results[0]['float_ADM_feature_adm_num_sasc_0_score']

    def test_run_float_adm_fextractor_dlm_weight_0d2(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_dlm_weight': 0.2}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_dlmw_0.2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_dlmw_0.2_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_dlmw_0.2_score'], 0.9656557708333334, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_dlmw_0.2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_dlmw_0.2_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_dlmw_0.2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_dlmw_0.2_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_dlmw_0.2_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_dlmw_0.2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_dlmw_0.2_score'], 0.9649663541666667, places=4)
        with self.assertRaises(KeyError):
            s = results[0]['float_ADM_feature_adm_num_dlmw_0.2_score']

    def test_run_float_adm_fextractor_dlm_weight_0d8(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_dlm_weight': 0.8}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_dlmw_0.8_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_dlmw_0.8_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_dlmw_0.8_score'], 0.9423001249999999, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_dlmw_0.8_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_dlmw_0.8_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_dlmw_0.8_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_dlmw_0.8_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_dlmw_0.8_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_dlmw_0.8_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_dlmw_0.8_score'], 0.9649663541666667, places=4)
        with self.assertRaises(KeyError):
            s = results[0]['float_ADM_feature_adm_num_dlmw_0.8_score']

    def test_run_float_adm_fextractor_adm_fs_1080_3h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_f1s0': 0.01183, 'adm_f1s1': 0.025026, 'adm_f1s2': 0.04295, 'adm_f1s3': 0.058621,
                           'adm_f2s0': 0.004302, 'adm_f2s1': 0.011778, 'adm_f2s2': 0.023918, 'adm_f2s3': 0.035901}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_f1s0_0.01183_f1s1_0.025026_f1s2_0.04295_f1s3_0.058621_f2s0_0.004302_f2s1_0.011778_f2s2_0.023918_f2s3_0.035901_score'], 0.9420788125, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_f1s0_0.01183_f1s1_0.025026_f1s2_0.04295_f1s3_0.058621_f2s0_0.004302_f2s1_0.011778_f2s2_0.023918_f2s3_0.035901_score'], 0.0218330625, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_f1s0_0.01183_f1s1_0.025026_f1s2_0.04295_f1s3_0.058621_f2s0_0.004302_f2s1_0.011778_f2s2_0.023918_f2s3_0.035901_score'], 0.9601228541666668, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_f1s0_0.01183_f1s1_0.025026_f1s2_0.04295_f1s3_0.058621_f2s0_0.004302_f2s1_0.011778_f2s2_0.023918_f2s3_0.035901_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_f1s0_0.01183_f1s1_0.025026_f1s2_0.04295_f1s3_0.058621_f2s0_0.004302_f2s1_0.011778_f2s2_0.023918_f2s3_0.035901_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_f1s0_0.01183_f1s1_0.025026_f1s2_0.04295_f1s3_0.058621_f2s0_0.004302_f2s1_0.011778_f2s2_0.023918_f2s3_0.035901_score'], 1.0, places=6)

    def test_run_float_adm_fextractor_adm_fs_1080_5h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_f1s0': 0.004212, 'adm_f1s1': 0.014809, 'adm_f1s2': 0.029642, 'adm_f1s3': 0.047464,
                           'adm_f2s0': 0.000984, 'adm_f2s1': 0.005852, 'adm_f2s2': 0.0146, 'adm_f2s3': 0.027574}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_f1s0_0.004212_f1s1_0.014809_f1s2_0.029642_f1s3_0.047464_f2s0_0.000984_f2s1_0.005852_f2s2_0.0146_f2s3_0.027574_score'], 0.9517253541666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_f1s0_0.004212_f1s1_0.014809_f1s2_0.029642_f1s3_0.047464_f2s0_0.000984_f2s1_0.005852_f2s2_0.0146_f2s3_0.027574_score'], 0.017266854166666668, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_f1s0_0.004212_f1s1_0.014809_f1s2_0.029642_f1s3_0.047464_f2s0_0.000984_f2s1_0.005852_f2s2_0.0146_f2s3_0.027574_score'], 0.9672292291666666, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_f1s0_0.004212_f1s1_0.014809_f1s2_0.029642_f1s3_0.047464_f2s0_0.000984_f2s1_0.005852_f2s2_0.0146_f2s3_0.027574_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_f1s0_0.004212_f1s1_0.014809_f1s2_0.029642_f1s3_0.047464_f2s0_0.000984_f2s1_0.005852_f2s2_0.0146_f2s3_0.027574_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_f1s0_0.004212_f1s1_0.014809_f1s2_0.029642_f1s3_0.047464_f2s0_0.000984_f2s1_0.005852_f2s2_0.0146_f2s3_0.027574_score'], 1.0, places=6)

    def test_run_float_adm_fextractor_adm_fs_2160_3h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_f1s0': 0.00226, 'adm_f1s1': 0.01183, 'adm_f1s2': 0.025026, 'adm_f1s3': 0.04295,
                           'adm_f2s0': 0.000479, 'adm_f2s1': 0.004302, 'adm_f2s2': 0.011778, 'adm_f2s3': 0.023918}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_f1s0_0.00226_f1s1_0.01183_f1s2_0.025026_f1s3_0.04295_f2s0_0.000479_f2s1_0.004302_f2s2_0.011778_f2s3_0.023918_score'], 0.9554477708333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_f1s0_0.00226_f1s1_0.01183_f1s2_0.025026_f1s3_0.04295_f2s0_0.000479_f2s1_0.004302_f2s2_0.011778_f2s3_0.023918_score'], 0.015557437499999998, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_f1s0_0.00226_f1s1_0.01183_f1s2_0.025026_f1s3_0.04295_f2s0_0.000479_f2s1_0.004302_f2s2_0.011778_f2s3_0.023918_score'], 0.9699451250000001, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_f1s0_0.00226_f1s1_0.01183_f1s2_0.025026_f1s3_0.04295_f2s0_0.000479_f2s1_0.004302_f2s2_0.011778_f2s3_0.023918_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_f1s0_0.00226_f1s1_0.01183_f1s2_0.025026_f1s3_0.04295_f2s0_0.000479_f2s1_0.004302_f2s2_0.011778_f2s3_0.023918_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_f1s0_0.00226_f1s1_0.01183_f1s2_0.025026_f1s3_0.04295_f2s0_0.000479_f2s1_0.004302_f2s2_0.011778_f2s3_0.023918_score'], 1.0, places=6)

    def test_run_float_adm_fextractor_adm_fs_2160_5h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_f1s0': 0.000092, 'adm_f1s1': 0.004212, 'adm_f1s2': 0.014809, 'adm_f1s3': 0.029642,
                           'adm_f2s0': 0.000050, 'adm_f2s1': 0.000984, 'adm_f2s2': 0.005852, 'adm_f2s3': 0.0146}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_f1s0_9.2e-05_f1s1_0.004212_f1s2_0.014809_f1s3_0.029642_f2s0_5e-05_f2s1_0.000984_f2s2_0.005852_f2s3_0.0146_score'], 0.9662835416666665, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_f1s0_9.2e-05_f1s1_0.004212_f1s2_0.014809_f1s3_0.029642_f2s0_5e-05_f2s1_0.000984_f2s2_0.005852_f2s3_0.0146_score'], 0.010868916666666666, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_f1s0_9.2e-05_f1s1_0.004212_f1s2_0.014809_f1s3_0.029642_f2s0_5e-05_f2s1_0.000984_f2s2_0.005852_f2s3_0.0146_score'], 0.9777072708333333, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_f1s0_9.2e-05_f1s1_0.004212_f1s2_0.014809_f1s3_0.029642_f2s0_5e-05_f2s1_0.000984_f2s2_0.005852_f2s3_0.0146_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_f1s0_9.2e-05_f1s1_0.004212_f1s2_0.014809_f1s3_0.029642_f2s0_5e-05_f2s1_0.000984_f2s2_0.005852_f2s3_0.0146_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_f1s0_9.2e-05_f1s1_0.004212_f1s2_0.014809_f1s3_0.029642_f2s0_5e-05_f2s1_0.000984_f2s2_0.005852_f2s3_0.0146_score'], 1.0, places=6)

    def test_run_float_adm_fextractor_barten_csf(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_1_scores'][0], 0.965404, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_1_score'], 0.9395831666666666, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_csf_1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_csf_1_scores'][0], 0.006402, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_csf_1_score'], 0.017910333333333334, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_csf_1_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_1_score'], 0.9608363333333333, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_csf_1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_csf_1_score'], 0.7635685833333333, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_csf_1_score'], 0.8480415833333333, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_csf_1_score'], 0.9165746666666666, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_csf_1_score'], 0.9612470208333335, places=4)

    def test_run_float_adm_fextractor_barten_csf_scale_coeffs(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_scores'][0], 0.972729, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9521435, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_scores'][0], 0.005644, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_score'], 0.015581145833333332, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9682811458333335, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9704822916666666, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9163563541666667, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_score'], 0.930392125, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9651669583333334, places=6)

    def test_run_float_adm_fextractor_barten_csf_scale_vhd_anw(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_noise_weight': 0.000625, 'adm_csf_scale': 0.002893,
                           'adm_csf_diag_scale': 0.001586}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_scores'][0], 0.969328, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 0.9461571666666666, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_scores'][0], 0.006347, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 0.017533479166666668, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 0.9643117916666667, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 0.9190820416666666, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 0.8794908124999999, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 0.9243578958333334, places=5)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nw_0.000625_score'], 0.9643775624999998, places=6)

    def test_run_float_adm_fextractor_barten_csf_scale_vhd_anw_0d5(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_noise_weight': 0.5, 'adm_csf_scale': 0.002893,
                           'adm_csf_diag_scale': 0.001586}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_scores'][0], 0.977851, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9611520625000001, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_scores'][0], 0.004584, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.012644520833333334, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9742538124999999, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9873047291666666, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9489389375, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9403206041666667, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9667059166666666, places=6)

    def test_run_float_adm_fextractor_barten_csf_scale_coeffs_1d5h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586, 'adm_norm_view_dist': 1.5}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_scores'][0], 0.962869, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9357588125, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9554831041666668, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9125230833333333, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.8846063333333333, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9267192083333334, places=5)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9651608125000001, places=6)

    def test_run_float_adm_fextractor_barten_csf_scale_coeffs_1d5h_rdh_2160(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586,
                           'adm_norm_view_dist': 1.5, 'adm_ref_display_height': 2160}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_scores'][0], 0.972729, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 0.9521435, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_scores'][0], 0.005644, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 0.015581145833333332, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 0.9682811458333335, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 0.9704822916666666, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 0.9163563541666667, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 0.930392125, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdf_2160_score'], 0.9651669583333334, places=6)

    def test_run_float_adm_fextractor_barten_csf_scale_coeffs_4d5h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586, 'adm_norm_view_dist': 4.5}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_scores'][0], 0.977101, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9595439791666666, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_scores'][0], 0.003975, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.012043541666666666, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.97375025, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.988183125, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9447696875, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9357725833333332, places=5)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9655332291666667, places=6)

    def test_run_float_adm_fextractor_barten_csf_scale_coeffs_4d5h_rdh_2160(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586,
                           'adm_norm_view_dist': 4.5, 'adm_ref_display_height': 2160}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_rdf_2160_scores'][0], 0.98511, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_rdf_2160_score'], 0.9734765, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_rdf_2160_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_rdf_2160_score'], 0.9832956666666668, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_rdf_2160_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_rdf_2160_score'], 0.9986556041666668, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_rdf_2160_score'], 0.9843173333333333, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_rdf_2160_score'], 0.9564127916666667, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_rdf_2160_score'], 0.9674029583333335, places=6)

    def test_run_float_adm_fextractor_barten_csf_scale_coeffs_1d5h_rdh_1080(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586,
                           'adm_norm_view_dist': 1.5, 'adm_ref_display_height': 1080}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_scores'][0], 0.962869, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9357588125, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9554831041666668, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9125230833333333, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.8846063333333333, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9267192083333334, places=5)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_score'], 0.9651608125000001, places=6)

    def test_run_float_adm_fextractor_barten_csf_scale_coeffs_2_3h_rdh_1080(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.1, 'adm_csf_diag_scale': 0.05,
                           'adm_norm_view_dist': 3, 'adm_ref_display_height': 1080}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.05_csf_1_scf_0.1_scores'][0], 0.968295, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.05_csf_1_scf_0.1_score'], 0.9443016041666666, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.05_csf_1_scf_0.1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.05_csf_1_scf_0.1_score'], 0.9629040833333334, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.05_csf_1_scf_0.1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.05_csf_1_scf_0.1_score'], 0.8088678749999999, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.05_csf_1_scf_0.1_score'], 0.8597765, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.05_csf_1_scf_0.1_score'], 0.9227575625000001, places=5)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.05_csf_1_scf_0.1_score'], 0.9644491666666667, places=6)

    def test_run_float_adm_fextractor_barten_csf_scale(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.5}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_1_scf_0.5_scores'][0], 0.961391, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_1_scf_0.5_score'], 0.9331726458333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_1_scf_0.5_scores'][0], 0.976321, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_1_scf_0.5_score'], 0.9558785833333333, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_csf_1_scf_0.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_csf_1_scf_0.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_csf_1_scf_0.5_score'], 0.7503370625, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_csf_1_scf_0.5_score'], 0.8354932708333335, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_csf_1_scf_0.5_score'], 0.9081349583333335, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_csf_1_scf_0.5_score'], 0.95661825, places=4)

    def test_run_float_adm_fextractor_barten_csf_scale_diag(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_diag_scale': 0.5}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.5_csf_1_scores'][0], 0.968146, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_scfd_0.5_csf_1_score'], 0.9440375625, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.5_csf_1_scores'][0], 0.980675, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_scfd_0.5_csf_1_score'], 0.9627281458333333, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_scfd_0.5_csf_1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_scfd_0.5_csf_1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_scfd_0.5_csf_1_score'], 0.775284, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_scfd_0.5_csf_1_score'], 0.8571071250000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_scfd_0.5_csf_1_score'], 0.922507375, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_scfd_0.5_csf_1_score'], 0.9644197083333333, places=4)

    def test_run_float_adm_fextractor_barten_csf_noise_weight(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_noise_weight': 0.00015625}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_1_nw_0.00015625_scores'][0], 0.965391, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_1_nw_0.00015625_score'], 0.9395595416666667, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_1_nw_0.00015625_scores'][0], 0.979493, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_1_nw_0.00015625_score'], 0.9608210208333334, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_csf_1_nw_0.00015625_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_csf_1_nw_0.00015625_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_csf_1_nw_0.00015625_score'], 0.7600680416666666, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_csf_1_nw_0.00015625_score'], 0.8478043541666667, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_csf_1_nw_0.00015625_score'], 0.9165525625000001, places=5)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_csf_1_nw_0.00015625_score'], 0.9612444166666666, places=6)

    def test_run_float_adm_fextractor_barten_csf_nvd_1d5(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_norm_view_dist': 1.5}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_1_nvd_1.5_scores'][0], 0.955282, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_1_nvd_1.5_score'], 0.922735875, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_1_nvd_1.5_scores'][0], 0.972047, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_1_nvd_1.5_score'], 0.9476545624999999, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_csf_1_nvd_1.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_csf_1_nvd_1.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_csf_1_nvd_1.5_score'], 0.7603636041666667, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_csf_1_nvd_1.5_score'], 0.8478539999999999, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_csf_1_nvd_1.5_score'], 0.9165624583333333, places=5)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_csf_1_nvd_1.5_score'], 0.9612470208333335, places=6)

    def test_run_float_adm_fextractor_adm_csf(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 2}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_2_scores'][0], 0.96851, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_2_score'], 0.9452254375, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_2_scores'][0], 0.981156, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_2_score'], 0.9641855833333333, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_csf_2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_csf_2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_csf_2_scores'][0], 0.97926, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_csf_2_score'], 0.9644948750000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_csf_2_score'], 0.8716099166666668, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_csf_2_score'], 0.9219838541666666, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_csf_2_score'], 0.9619686041666666, places=4)

    def test_run_float_adm_fextractor_adm_bypass_cm(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_bypass_cm': 1}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_bcm_1_scores'][0], 0.973175, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_bcm_1_score'], 0.9513183749999999, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_bcm_1_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_bcm_1_scores'][0], 0.958174, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_bcm_1_score'], 0.9229132916666666, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_bcm_1_score'], 0.9172484583333332, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_bcm_1_score'], 0.9495412291666666, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_bcm_1_score'], 0.9773672916666666, places=4)

    def test_run_float_adm_fextractor_adm_p_norm(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_p_norm': 3.5}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_apn_3.5_scores'][0], 0.968458, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_apn_3.5_score'], 0.9446595208333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_apn_3.5_scores'][0], 0.978242, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_apn_3.5_score'], 0.9580824583333333, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_apn_3.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_apn_3.5_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_apn_3.5_scores'][0], 0.953791, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_apn_3.5_score'], 0.9192486666666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_apn_3.5_score'], 0.9067780833333333, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_apn_3.5_score'], 0.9399117291666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_apn_3.5_score'], 0.9702566666666667, places=4)

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
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_score'], 0.9539779375, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_score'], 1.0, places=6)
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
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_score'], 0.9539779375, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_score'], 0.9649663541666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_score'], 0.9345148541666667, places=4)

        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_num_score'], 371.80645372916666, places=3)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_den_score'], 397.83379106250004, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_num_scale0_score'], 45.526146958333335, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_den_scale0_score'], 50.14385129166667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_num_scale1_score'], 66.574236, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_den_scale1_score'], 74.47438383333333, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_num_scale2_score'], 105.55483329166668, places=3)
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
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_score'], 0.011558, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_score'], 1.052564, places=6)

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
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_egl_1_score'], 0.9574308606115118, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_egl_1_score'], 0.013683, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_egl_1_score'], 0.971874, places=6)

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
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_egl_1.2_score'], 0.01157, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_egl_1.2_score'], 1.052512, places=6)

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
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_score'], 1.052366, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_score'], 1.070267, places=5)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_score'], 1.072423, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_score'], 1.072756, places=4)

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
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale0_egl_1_score'], 0.983708, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale1_egl_1_score'], 0.997443, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale2_egl_1_score'], 0.998483, places=4)
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_egl_1_score'], 0.999151, places=4)

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
        self.assertAlmostEqual(results[0]['float_VIF_feature_vif_scale3_egl_1.1_score'], 1.04892, places=3)

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
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale0_egl_1_score'], 0.983708, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale1_egl_1_score'], 0.997443, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale2_egl_1_score'], 0.998483, places=4)
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_egl_1_score'], 0.999151, places=4)

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
        self.assertAlmostEqual(results[0]['integer_VIF_feature_vif_scale3_egl_1.1_score'], 1.04892, places=3)

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
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_score'], 0.026560104166666664, places=4)  # float 0.026559020833333336
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_score'], 0.9539728125, places=4)  # float 0.9539779375
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_score'], 1.000002, places=6)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_score'], 1.000001, places=6)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_score'], 0.9649663541666667, places=4)
        with self.assertRaises(KeyError):
            s = results[0]['integer_adm_num_score']

    def test_run_integer_adm_fextractor_skip_aim(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_skip_aim': True},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_score'], 0.9345057916666667, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_score'], 0.0, places=4)
        # adm3 is not same as adm2 here, because when skipping AIM calculation, AIM becomes 0.
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_score'], 0.9672529583333332, places=4)

    def test_run_integer_adm_fextractor_barten_watson_blend_1080_3h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 2, 'adm_norm_view_dist': 3.0, 'adm_ref_display_height': 1080},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_2_score'], 0.9420671875000001, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_2_score'], 0.021833541666666664, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_csf_2_score'], 0.9601167708333334, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_csf_2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_csf_2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_csf_2_score'], 1.0, places=4)

    def test_run_integer_adm_fextractor_barten_watson_blend_1080_5h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 2, 'adm_norm_view_dist': 5.0, 'adm_ref_display_height': 1080},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_2_nvd_5_score'], 0.9517139375000001, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_2_nvd_5_score'], 0.0172674375, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_csf_2_nvd_5_score'], 0.96722325, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_csf_2_nvd_5_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_csf_2_nvd_5_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_csf_2_nvd_5_score'], 1.0, places=4)

    def test_run_integer_adm_fextractor_barten_watson_blend_2160_3h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 2, 'adm_norm_view_dist': 3.0, 'adm_ref_display_height': 2160},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_2_rdh_2160_score'], 0.95543625, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_2_rdh_2160_score'], 0.015557937499999999, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_csf_2_rdh_2160_score'], 0.9699390833333333, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_csf_2_rdh_2160_score'], 0.9999995624999999, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_csf_2_rdh_2160_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_csf_2_rdh_2160_score'], 1.0, places=4)

    def test_run_integer_adm_fextractor_barten_watson_blend_2160_5h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 2, 'adm_norm_view_dist': 5.0, 'adm_ref_display_height': 2160},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_2_nvd_5_rdh_2160_score'], 0.9662732083333333, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_2_nvd_5_rdh_2160_score'], 0.010870229166666667, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_csf_2_nvd_5_rdh_2160_score'], 0.9777014583333333, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_csf_2_nvd_5_rdh_2160_score'], 0.9999995624999999, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_csf_2_nvd_5_rdh_2160_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_csf_2_nvd_5_rdh_2160_score'], 1.0, places=4)

    def test_run_integer_adm_fextractor_barten_csf_scale_coeffs(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_scores'][0], 0.972724, places=6)  # float 0.972729
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9521308541666667, places=6)  # float 0.9521435
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_scores'][0], 0.005646, places=6)  # float 0.005644
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_score'], 0.015581666666666667, places=6)  # float 0.015581145833333332
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9682745416666666, places=6)  # float 0.9682811458333335
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=6)  # float 1.0
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=6)  # float 1.0
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9704815416666666, places=6)  # float 0.9704822916666666
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9163508125000001, places=6)  # float 0.9163563541666667
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9303777083333333, places=6)  # float 0.930392125
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9651512916666666, places=6)  # float 0.9651669583333334

    def test_run_integer_adm_fextractor_barten_csf_scale_coeffs_change_adm_nw(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586,
                           'adm_noise_weight': 0.5},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_scores'][0], 0.977847, places=6)  # float 0.977851
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9611413333333333, places=6)  # float 0.9611520625000001
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_scores'][0], 0.004585, places=6)  # float 0.004584
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.012645187500000002, places=6)  # float 0.012644520833333334
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9742480416666668, places=6)  # float 0.9742538124999999
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 1.0, places=6)  # float 1.0
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 1.0, places=6)  # float 1.0
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9873043541666666, places=6)  # float 0.9873047291666666
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9489353125000001, places=6)  # float 0.9489389375
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9403079791666666, places=6)  # float 0.9403206041666667
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_score'], 0.9666907916666667, places=6)  # float 0.9667059166666666

    def test_run_integer_adm_fextractor_barten_csf_scale_coeffs_4d5h(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586, 'adm_norm_view_dist': 4.5},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_scores'][0], 0.977096, places=6)  # float 0.977101
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9595311250000002, places=6)  # float 0.9595439791666666
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_scores'][0], 0.003978, places=6)  # float 0.003975
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.012044041666666666, places=6)  # float 0.012043541666666666
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9737435833333333, places=6)  # float 0.97375025
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.999999, places=6)  # float 1.0
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.999999125, places=6)  # float 1.0
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9881750416666666, places=6)  # float 0.988183125
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9447661041666665, places=6)  # float 0.9447696875
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9357595833333333, places=6)  # float 0.9357725833333332
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9655177708333333, places=6)  # float 0.9655332291666667

    def test_run_integer_adm_fextractor_barten_csf_scale_coeffs_1d5h_rdh_2160(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586,
                           'adm_norm_view_dist': 1.5, 'adm_ref_display_height': 2160},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_scores'][0], 0.972724, places=6)  # float 0.972729
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_score'], 0.9521308541666667, places=6)  # float 0.9521435
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_scores'][0], 0.005646, places=6)  # float 0.005644
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_score'], 0.015581666666666667, places=6)  # float 0.015581145833333332
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_score'], 0.9682745416666666, places=6)  # float 0.9682811458333335
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_score'], 1.0, places=6)  # float 1.0
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_score'], 1.0, places=6)  # float 1.0
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_score'], 0.9704815416666666, places=6)  # float 0.9704822916666666
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_score'], 0.9163508125000001, places=6)  # float 0.9163563541666667
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_score'], 0.9303777083333333, places=6)  # float 0.930392125
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_nvd_1.5_rdh_2160_score'], 0.9651512916666666, places=6)  # float 0.9651669583333334

    def test_run_integer_adm_fextractor_dlmw(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_dlm_weight': 0.8},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_dlmw_0.8_score'], 0.9345057916666667, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_dlmw_0.8_score'], 0.026560104166666664, places=4)  # float 0.026559020833333336
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_dlmw_0.8_score'], 0.9422925833333334, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_dlmw_0.8_score'], 1.000002, places=6)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_dlmw_0.8_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_dlmw_0.8_score'], 1.000002, places=6)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_dlmw_0.8_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_dlmw_0.8_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_dlmw_0.8_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_dlmw_0.8_score'], 0.9649663541666667, places=4)

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
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_score'], 0.026560104166666664, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_score'], 0.9539728125, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_score'], 1.000002, places=6)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_score'], 1.000001, places=6)
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
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_score'], 0.018717666666666664, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_score'], 0.9665263333333334, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_score'], 1.000002, places=6)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_score'], 0.0, places=6)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_score'], 1.000001, places=6)

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
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_score'], 0.011556, places=5)  # float 0.011558
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_score'], 1.052572, places=5)  # float 1.052564

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
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_egl_1_score'], 0.013684, places=6)  # float 0.013683
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_egl_1_score'], 0.971875, places=5)  # float 0.971874

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
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_egl_1.2_score'], 0.01157, places=5)  # float 0.01157
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_egl_1.2_score'], 1.05252, places=5)  # float 1.052512

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

    def test_run_float_adm_fextractor_v1017_csf_basic(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 3}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_3_scores'][0], 0.962086, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_3_score'], 0.9345148750000001, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_csf_3_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_csf_3_scores'][0], 0.011381, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_csf_3_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_csf_3_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_3_score'], 0.9539779375, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_csf_3_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_csf_3_score'], 0.9078873333333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_csf_3_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_csf_3_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_csf_3_score'], 0.9649663541666667, places=4)

    def test_run_float_adm_fextractor_v1017_csf_3d0h_1080(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={
                'adm_csf_mode': 3,
                'adm_dlm_weight': 0.7,
                'adm_noise_weight': 0.02,
                'adm_norm_view_dist': 3.0,
                'adm_ref_display_height': 1080,
                'adm_min_val': 0.5
            }
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_scores'][0], 0.961162, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.9329204583333333, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_scores'][0], 0.011659, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.027206229166666665, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.9448824583333334, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.8997115, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.8900255833333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.9289640208333334, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.9647255625, places=4)

    def test_run_float_adm_fextractor_v1017_csf_3d0h_2160(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = FloatAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={
                'adm_csf_mode': 3,
                'adm_dlm_weight': 0.7,
                'adm_noise_weight': 0.02,
                'adm_norm_view_dist': 3.0,
                'adm_ref_display_height': 2160,
                'adm_min_val': 0.5
            }
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_scores'][0], 0.971938, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 0.9513279166666667, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_scores'][0], 0.007912, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 0.019080583333333335, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm3_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 0.9602053333333332, places=4)
        self.assertAlmostEqual(results[1]['float_ADM_feature_adm3_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 1.0, places=6)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale0_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 0.959960375, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale1_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 0.925689375, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale2_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 0.9387684166666667, places=4)
        self.assertAlmostEqual(results[0]['float_ADM_feature_adm_scale3_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdf_2160_score'], 0.9669660000000001, places=4)

    def test_run_integer_adm_fextractor_v1017_csf_basic(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 3}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_3_scores'][0], 0.96625, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_3_score'], 0.9414857083333334, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_csf_3_score'], 1.0, places=5)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_3_scores'][0], 0.009258, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_3_score'], 0.022306458333333334, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_csf_3_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_csf_3_score'], 0.9595895833333333, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_csf_3_score'], 1.0, places=5)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_csf_3_score'], 0.9297658958333334, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_csf_3_score'], 0.9031356458333333, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_csf_3_score'], 0.9310575, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_csf_3_score'], 0.9649511666666667, places=4)

    def test_run_integer_adm_fextractor_v1017_csf_3d0h_1080(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={
                'adm_csf_mode': 3,
                'adm_dlm_weight': 0.7,
                'adm_noise_weight': 0.02,
                'adm_norm_view_dist': 3.0,
                'adm_ref_display_height': 1080,
                'adm_min_val': 0.5
            }
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_scores'][0], 0.965334, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.9398977916666666, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 1.0, places=5)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_scores'][0], 0.009509, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.02291233333333333, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.9510548124999999, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 1.0, places=5)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.922381, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.8986252708333332, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.9298436041666668, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_csf_3_dlmw_0.7_min_0.5_nw_0.02_score'], 0.9647255625, places=4)

    def test_run_integer_adm_fextractor_v1017_csf_3d0h_2160(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = IntegerAdmFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={
                'adm_csf_mode': 3,
                'adm_dlm_weight': 0.7,
                'adm_noise_weight': 0.02,
                'adm_norm_view_dist': 3.0,
                'adm_ref_display_height': 2160,
                'adm_min_val': 0.5
            }
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_scores'][0], 0.973633, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 0.9539956875, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm2_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 1.0, places=5)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_scores'][0], 0.006305, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 0.016282354166666662, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_aim_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 0.0, places=6)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm3_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 0.9629122708333332, places=4)
        self.assertAlmostEqual(results[1]['integer_ADM_feature_adm3_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 1.0, places=5)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale0_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 0.9804549791666665, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale1_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 0.9241072291666669, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale2_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 0.9356613125000001, places=4)
        self.assertAlmostEqual(results[0]['integer_ADM_feature_adm_scale3_csf_3_dlmw_0.7_min_0.5_nw_0.02_rdh_2160_score'], 0.9657157916666668, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
