import os
import shutil
import unittest
from test.testutil import set_default_576_324_videos_for_testing, \
    set_default_576_324_videos_for_testing_scaled, \
    set_default_cambi_video_for_testing_b, \
    set_default_cambi_video_for_testing_10b, \
    set_default_cambi_video_for_testing_mp4

from vmaf.core.cambi_feature_extractor import CambiFeatureExtractor, CambiFullReferenceFeatureExtractor
from vmaf.core.cambi_quality_runner import CambiQualityRunner, CambiFullReferenceQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from vmaf.tools.misc import MyTestCase


class CambiFeatureExtractorTest(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_run_cambi_fextractor(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_8_score'],
                               0.25968416666666666, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_encbd_8_score'],
                               0.00020847916666666663, places=4)

    def test_run_cambi_fextractor_topk(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'cambi_topk': 0.7}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_ctpk_0.7_encbd_8_score'],
                               0.22258620833333334, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_ctpk_0.7_encbd_8_score'],
                               0.00020847916666666663, places=4)

    def test_run_cambi_fextractor_both_topk_set(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'topk': 0.8,
                           'cambi_topk': 0.7}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_ctpk_0.7_encbd_8_topk_0.8_score'],
                               0.1947626041666667, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_ctpk_0.7_encbd_8_topk_0.8_score'],
                               0.00020847916666666663, places=4)

    def test_run_cambi_fextractor_eotf(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'cambi_eotf': "pq"}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_ceot_pq_encbd_8_score'],
                               0.2596885625, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_ceot_pq_encbd_8_score'],
                               0.00020847916666666663, places=4)

    def test_run_cambi_fextractor_both_eotf_set(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'eotf': "xxx",
                           'cambi_eotf': "pq"}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_ceot_pq_encbd_8_eotf_xxx_score'],
                               0.2596885625, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_ceot_pq_encbd_8_eotf_xxx_score'],
                               0.00020847916666666663, places=4)

    def test_run_cambi_fextractor_scaled(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_scaled()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_8_ench_270_encw_480_score'],
                               0.17871631249999997, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_encbd_8_ench_270_encw_480_score'],
                               0.00022027083333333336, places=4)

    def test_run_cambi_fextractor_scaled_b(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_b()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_8_ench_540_encw_960_score'],
                               0.773078, places=4)

    def test_run_cambi_fextractor_10b(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_10b()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_10_score'],
                               0.0013863333333333334, places=4)

    def test_run_cambi_fextractor_notyuv_unspecified_enc_bitdepth(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        with self.assertRaises(AssertionError):
            self.fextractor.run(parallelize=False)

    def test_run_cambi_fextractor_notyuv_correct_enc_bitdepth(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        asset.asset_dict['dis_enc_bitdepth'] = 8
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_8_score'],
                               0.022446666666666667, places=4)

    def test_run_cambi_fextractor_notyuv_incorrect_enc_bitdepth(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        asset.asset_dict['dis_enc_bitdepth'] = 10
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_10_score'],
                               0.020136333333333336, places=4)

    def test_run_cambi_fextractor_notyuv_10bit_without_workfile_yuv_type(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        asset.asset_dict['dis_enc_bitdepth'] = 10
        del asset.asset_dict['workfile_yuv_type']
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )

        with self.assertRaises(AssertionError):
            self.fextractor.run(parallelize=False)

    def test_run_cambi_fextractor_max_log_contrast(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'max_log_contrast': 4}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_8_mlc_4_score'],
                               0.32247770833333333, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_encbd_8_mlc_4_score'],
                               0.0002582708333333333, places=4)

        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'max_log_contrast': 0}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_8_mlc_0_score'],
                               0.0024372291666666665, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_encbd_8_mlc_0_score'],
                               0.00019045833333333336, places=4)

    def test_run_cambi_fextractor_full_reference(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFullReferenceFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_encbd_8_score'],
                               0.25968416666666666, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_source_score'],
                               0.00020847916666666663, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_full_reference_score'],
                               0.2594353333333333, places=4)

    def test_run_cambi_fextractor_full_reference_scaled_ref(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFullReferenceFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'src_width': 480, 'src_height': 270}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_encbd_8_srch_270_srcw_480_score'],
                               0.25968416666666666, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_source_score'],
                               0.00022027083333333336, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_full_reference_score'],
                               0.25944045833333335, places=4)

    def test_run_cambi_fextractor_full_reference_scaled_ref_max_val(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFullReferenceFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'src_width': 480, 'src_height': 270, 'cambi_max_val': 0.0001}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_cmxv_0.0001_encbd_8_srch_270_srcw_480_score'],
                               0.0001, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_source_score'], 0.0001, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_full_reference_score'], 0.0001, places=4)

    def test_run_cambi_fextractor_notyuv_4k_encode(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        asset.asset_dict['dis_enc_bitdepth'] = 8
        asset.asset_dict['quality_width'] = 3840
        asset.asset_dict['quality_height'] = 2160
        asset.asset_dict['dis_enc_width'] = 3840
        asset.asset_dict['dis_enc_height'] = 2160

        self.fextractor = CambiFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_8_score'], 0.24399833333333332, places=4)

    def test_run_cambi_fextractor_notyuv_4k_encode_high_res_speedup(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        asset.asset_dict['dis_enc_bitdepth'] = 8
        asset.asset_dict['quality_width'] = 3840
        asset.asset_dict['quality_height'] = 2160
        asset.asset_dict['dis_enc_width'] = 3840
        asset.asset_dict['dis_enc_height'] = 2160

        self.fextractor = CambiFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'cambi_high_res_speedup': 2160}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_hrs_2160_encbd_8_score'], 0.24772933333333333, places=4)

    def test_run_cambi_fextractor_notyuv_1080p_encode(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        asset.asset_dict['dis_enc_bitdepth'] = 8
        asset.asset_dict['quality_width'] = 3840
        asset.asset_dict['quality_height'] = 2160
        asset.asset_dict['dis_enc_width'] = 1920
        asset.asset_dict['dis_enc_height'] = 1080

        self.fextractor = CambiFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_8_ench_1080_encw_1920_score'], 0.14994066666666664, places=4)

    def test_run_cambi_fextractor_notyuv_1080p_encode_high_res_speedup(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        asset.asset_dict['dis_enc_bitdepth'] = 8
        asset.asset_dict['quality_width'] = 3840
        asset.asset_dict['quality_height'] = 2160
        asset.asset_dict['dis_enc_width'] = 1920
        asset.asset_dict['dis_enc_height'] = 1080

        self.fextractor = CambiFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'cambi_high_res_speedup': 1080}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_hrs_1080_encbd_8_ench_1080_encw_1920_score'], 0.16270733333333334, places=4)

    def test_run_cambi_fextractor_notyuv_1080p_encode_vis_lum_threshold(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        asset.asset_dict['dis_enc_bitdepth'] = 8
        asset.asset_dict['quality_width'] = 3840
        asset.asset_dict['quality_height'] = 2160
        asset.asset_dict['dis_enc_width'] = 1920
        asset.asset_dict['dis_enc_height'] = 1080

        self.fextractor = CambiFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'cambi_vis_lum_threshold': 5.0}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_vlt_5_encbd_8_ench_1080_encw_1920_score'], 0.095409,
                               places=4)

    def test_run_cambi_fextractor_max_val(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'cambi_max_val': 0.1}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_cmxv_0.1_encbd_8_score'], 0.1, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_cmxv_0.1_encbd_8_score'], 0.0002582708333333333,
                               places=4)

    def test_run_cambi_fextractor_notyuv_1440p_encode_at_1080p_quality_height(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_mp4()
        asset.asset_dict['dis_enc_bitdepth'] = 8
        asset.asset_dict['quality_width'] = 1920
        asset.asset_dict['quality_height'] = 1080
        asset.asset_dict['dis_enc_width'] = 2560
        asset.asset_dict['dis_enc_height'] = 1440

        self.fextractor = CambiFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_encbd_8_ench_1440_encw_2560_score'], 0.15102933333333332, places=4)


class CambiQualityRunnerTest(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'qrunner'):
            self.qrunner.remove_results()
        super().tearDown()

    def test_run_cambi_runner(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.qrunner = CambiQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.qrunner.run(parallelize=True)
        results = self.qrunner.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_score'],
                               0.25968416666666666, places=4)
        self.assertAlmostEqual(results[1]['Cambi_score'],
                               0.00020847916666666663, places=4)

    def test_run_cambi_runner_scale(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_scaled()
        self.qrunner = CambiQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.qrunner.run(parallelize=True)
        results = self.qrunner.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_score'],
                               0.17871631249999997, places=4)
        self.assertAlmostEqual(results[1]['Cambi_score'],
                               0.00022027083333333336, places=4)

    def test_run_cambi_runner_scale_b(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_b()
        self.qrunner = CambiQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.qrunner.run(parallelize=True)
        results = self.qrunner.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_score'],
                               0.773078, places=4)

    def test_run_cambi_runner_10b(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_10b()
        self.qrunner = CambiQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.qrunner.run(parallelize=True)
        results = self.qrunner.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_score'],
                               0.0013863333333333334, places=4)

    def test_run_cambi_runner_fullref(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.qrunner = CambiFullReferenceQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
        )
        self.qrunner.run(parallelize=True)
        results = self.qrunner.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_FR_score'],
                               0.2594353333333333, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_score'],
                               0.25968416666666666, places=4)


class CambiResultsCachingTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.results_store_dir = FileSystemResultStore()

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        super().tearDown()

    def test_run_cambi_runner(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_scaled()
        self.qrunner = CambiQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=self.results_store_dir,
            optional_dict={}
        )

        # make sure the caching directory needs to be created by running the feature extractor
        fextractor= self.qrunner._get_feature_extractor_class()
        self.store_dir = os.path.join(
            self.results_store_dir.result_store_dir, f'{fextractor.TYPE}_V{fextractor.VERSION}', 'test')
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

        self.qrunner.run(parallelize=False)
        results = self.qrunner.results

        # check if the correct directory was created
        self.assertTrue(os.path.exists(self.store_dir))

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_score'],
                               0.17871631249999997, places=4)
        self.assertAlmostEqual(results[1]['Cambi_score'],
                               0.00022027083333333336, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)