import os
import shutil
import unittest
from test.testutil import set_default_576_324_videos_for_testing, \
    set_default_576_324_videos_for_testing_scaled, \
    set_default_cambi_video_for_testing_b, \
    set_default_cambi_video_for_testing_10b

from vmaf.core.cambi_feature_extractor import CambiFeatureExtractor, CambiFullReferenceFeatureExtractor
from vmaf.core.cambi_quality_runner import CambiQualityRunner, CambiFullReferenceQualityRunner
from vmaf.tools.misc import MyTestCase
from vmaf.core.result_store import FileSystemResultStore


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
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_score'],
                               0.25968416666666666, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_score'],
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
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_score'],
                               0.17871631249999997, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_score'],
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
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_score'],
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
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_score'],
                               0.0013863333333333334, places=4)

    def test_run_cambi_fextractor_incorrect_enc_bitdepth(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_10b()
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
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_score'],
                               0.0013863333333333334, places=4)

    def test_run_cambi_fextractor_correct_enc_bitdepth(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_10b()
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
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_score'],
                               0.00020733333333333332, places=4)

    def test_run_cambi_fextractor_notyuv_correct_enc_bitdepth_8(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.fextractor = CambiFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=False)
        results = self.fextractor.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_score'],
                               0.00020733333333333332, places=4)

    def test_run_cambi_fextractor_enc_bitdepth_none(self):
        _, _, asset, asset_original = set_default_cambi_video_for_testing_10b()
        asset.asset_dict['dis_enc_bitdepth'] = None
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
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_score'],
                               0.32247770833333333, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_score'],
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
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_score'],
                               0.0024372291666666665, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_score'],
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
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_score'],
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
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_score'],
                               0.25968416666666666, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_source_score'],
                               0.00022027083333333336, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_full_reference_score'],
                               0.25944045833333335, places=4)


class CambiQualityRunnerTest(MyTestCase):

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
        self.results_store_dir = FileSystemResultStore()

    def tearDown(self):
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

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