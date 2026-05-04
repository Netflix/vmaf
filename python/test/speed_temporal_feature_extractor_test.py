import unittest
from test.testutil import set_default_576_324_videos_for_testing_5frames

from vmaf.core.feature_extractor import SpeedTemporalFeatureExtractor
from vmaf.tools.misc import MyTestCase


class SpeedTemporalFeatureExtractorTest(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_run_speed_temporal_fextractor(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedTemporalFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_temporal_feature_speed_temporal_score'], 18.9411808, places=4)
        self.assertAlmostEqual(results[1]['Speed_temporal_feature_speed_temporal_score'], 0.0, places=4)

    def test_run_speed_temporal_fextractor_use_ref_diff(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.fextractor = SpeedTemporalFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'speed_use_ref_diff': True}
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Speed_temporal_feature_speed_temporal_urd_score'], 16.721622000000004, places=4)
        self.assertAlmostEqual(results[1]['Speed_temporal_feature_speed_temporal_urd_score'], 0.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
