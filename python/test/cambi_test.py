import unittest
from test.testutil import set_default_576_324_videos_for_testing, \
    set_default_576_324_videos_for_testing_scaled, \
    set_default_cambi_video_for_testing_b, \
    set_default_cambi_video_for_testing_10b

from vmaf.core.cambi_feature_extractor import CambiFeatureExtractor, CambiFullReferenceFeatureExtractor
from vmaf.core.cambi_quality_runner import CambiQualityRunner, CambiFullReferenceQualityRunner
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
        self.assertAlmostEqual(results[0]['Cambi_feature_cambi_score'],
                               0.7308609375000001, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_score'],
                               0.0014658541666666667, places=4)

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
                               0.9712994583333333, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_score'],
                               0.004251791666666667, places=4)

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
                               1.387623, places=4)

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
                               0.01497266666, places=4)

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
                               0.9576131666666666, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_score'],
                               0.0024499791666667, places=4)

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
                               0.017304916666666666, places=4)
        self.assertAlmostEqual(results[1]['Cambi_feature_cambi_score'],
                               0.000671125, places=4)

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
                               0.7308609375, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_source_score'],
                               0.00146585416, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_full_reference_score'],
                               0.72939404166, places=4)

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
                               0.7308609375, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_source_score'],
                               0.0042517916, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_full_reference_score'],
                               0.7265804791, places=4)


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
                               0.7308609375000001, places=4)
        self.assertAlmostEqual(results[1]['Cambi_score'],
                               0.0014658541666666667, places=4)

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
                               0.9712994583333333, places=4)
        self.assertAlmostEqual(results[1]['Cambi_score'],
                               0.004251791666666667, places=4)

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
                               1.387623, places=4)

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
                               0.014972666666666667, places=4)

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
                               0.7293940416666668, places=4)
        self.assertAlmostEqual(results[0]['Cambi_FR_feature_cambi_score'],
                               0.7308609375000001, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)