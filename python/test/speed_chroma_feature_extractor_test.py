import unittest
from test.testutil import set_default_576_324_videos_for_testing_5frames, \
    set_default_speed_chroma_edge_case, set_default_speed_chroma_edge_case_swapped

from vmaf.core.feature_extractor import SpeedChromaFeatureExtractor
from vmaf.tools.misc import MyTestCase


class SpeedChromaFeatureExtractorTest(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_run_speed_chroma_fextractor(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_score'], 11.7757982, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_score'], 3.8896207999999994, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_score'], 7.8327095, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_score'], 0.0, places=4)

    def test_run_speed_chroma_fextractor_mxv_1(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_max_val": 1},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_mxv_1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_mxv_1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_mxv_1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_mxv_1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_mxv_1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_mxv_1_score'], 0.0, places=4)

    def test_run_speed_chroma_fextractor_wrv(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_weight_var_mode": 1},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_wvm_1_score'], 3.5695335999999998, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_wvm_1_score'], 0.8904534, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_wvm_1_score'], 2.2299936000000002, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_wvm_1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_wvm_1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_wvm_1_score'], 0.0, places=4)

    def test_run_speed_chroma_ks1o2(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 0.5},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_0.5_score'], 13.807969, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_0.5_score'], 6.684846, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_0.5_score'], 10.246407, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_0.5_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_0.5_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_0.5_score'], 0.0, places=4)

    def test_run_speed_chroma_ks1o2d25(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 0.444444},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_0.444444_score'], 13.807969, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_0.444444_score'], 6.684846, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_0.444444_score'], 10.246407, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_0.444444_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_0.444444_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_0.444444_score'], 0.0, places=4)

    def test_run_speed_chroma_ks3o2(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 1.5},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_1.5_score'], 5.116476, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_1.5_score'], 6.029049, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_1.5_score'], 5.572762, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_1.5_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_1.5_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_1.5_score'], 0.0, places=4)

    def test_run_speed_chroma_ks2(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 2},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_2_score'], 5.524980, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_2_score'], 7.472359, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_2_score'], 6.498670, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_2_score'], 0.0, places=4)

    def test_run_speed_chroma_ks2o3(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 0.6666},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_0.6666_score'], 11.730496, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_0.6666_score'], 4.418727, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_0.6666_score'], 8.074611, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_0.6666_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_0.6666_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_0.6666_score'], 0.0, places=4)

    def test_run_speed_chroma_ks24o10(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 2.4},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_2.4_score'], 5.287785, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_2.4_score'], 7.663369, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_2.4_score'], 6.475577, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_2.4_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_2.4_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_2.4_score'], 0.0, places=4)

    def test_run_speed_chroma_ks360o97(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 3.711},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_3.711_score'], 6.075947, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_3.711_score'], 4.2133896, places=3)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_3.711_score'], 5.144629, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_3.711_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_3.711_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_3.711_score'], 0.0, places=4)

    def test_run_speed_chroma_ks4o3(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 1.3333},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_1.3333_score'], 8.113384, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_1.3333_score'], 5.146757, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_1.3333_score'], 6.630071, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_1.3333_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_1.3333_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_1.3333_score'], 0.0, places=4)

    def test_run_speed_chroma_ks3d5o3(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 1.1666},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_1.1666_score'], 10.722349, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_1.1666_score'], 4.989062, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_1.1666_score'], 7.855705, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_1.1666_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_1.1666_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_1.1666_score'], 0.0, places=4)

    def test_run_speed_chroma_ks3d75o3(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 1.25},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_1.25_score'], 8.113384, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_1.25_score'], 5.146757, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_1.25_score'], 6.630071, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_1.25_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_1.25_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_1.25_score'], 0.0, places=4)

    def test_run_speed_chroma_ks4d25o3(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 1.4166},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_1.4166_score'], 8.113384, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_1.4166_score'], 5.146757, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_1.4166_score'], 6.630071, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_1.4166_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_1.4166_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_1.4166_score'], 0.0, places=4)

    def test_run_speed_chroma_ks5o3(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 1.6666},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_1.6666_score'], 5.591870, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_1.6666_score'], 7.741881, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_1.6666_score'], 6.666876, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_1.6666_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_1.6666_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_1.6666_score'], 0.0, places=4)

    def test_run_speed_chroma_ks3(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 3.0000},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_3_score'], 5.741959, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_3_score'], 6.081584, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_3_score'], 5.911772, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_3_score'], 0.0, places=4)

    def test_run_speed_chroma_ks0d740740(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 0.740740},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_0.74074_score'], 11.730499400000001, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_0.74074_score'], 4.4187254000000005, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_0.74074_score'], 8.074612400000001, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_0.74074_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_0.74074_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_0.74074_score'], 0.0, places=4)

    def test_run_speed_chroma_ks1d111111(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_kernelscale": 1.111111},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ks_1.11111_score'], 10.722351799999998, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ks_1.11111_score'], 4.9890712, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ks_1.11111_score'], 7.8557114, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ks_1.11111_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ks_1.11111_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ks_1.11111_score'], 0.0, places=4)

    def test_run_speed_chroma_fs1(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_prescale": 1.0000},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_score'], 11.7757982, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_score'], 3.8896207999999994, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_score'], 7.8327095, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_score'], 0.0, places=4)

    def test_run_speed_chroma_fs2(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_prescale": 2.0000},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_ps_2_score'], 15.797304089929252, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_ps_2_score'], 22.887580115662352, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_ps_2_score'], 19.3424421027958, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_ps_2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_ps_2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_ps_2_score'], 0.0, places=4)

    def test_run_speed_chroma_sigma_nn_0d5(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_sigma_nn": 0.5},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_snn_0.5_score'], 11.7407674, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_snn_0.5_score'], 3.9838720000000003, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_snn_0.5_score'], 7.862319599999999, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_snn_0.5_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_snn_0.5_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_snn_0.5_score'], 0.0, places=4)

    def test_run_speed_chroma_ref_regular_dis_singular(self):
        _, _, asset, asset_original = set_default_speed_chroma_edge_case()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_score'], 5.514892, places=3)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_score'], 5.514892, places=3)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_score'], 0.0, places=4)

    def test_run_speed_chroma_ref_singular_dis_regular(self):
        _, _, asset, asset_original = set_default_speed_chroma_edge_case_swapped()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_score'], 5.514892, places=3)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_score'], 5.514892, places=3)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_score'], 0.0, places=4)

    def test_run_speed_chroma_nn_floor_0d1(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedChromaFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={"speed_nn_floor": 0.1},
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_u_nnf_0.1_score'], 11.775814400000002, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_v_nnf_0.1_score'], 3.889633000000001, places=4)
        self.assertAlmostEqual(results[0]['Speed_chroma_feature_speed_chroma_uv_nnf_0.1_score'], 7.832723799999999, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_u_nnf_0.1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_v_nnf_0.1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['Speed_chroma_feature_speed_chroma_uv_nnf_0.1_score'], 0.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
