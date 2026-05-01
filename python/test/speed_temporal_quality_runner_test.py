import unittest
from test.testutil import set_default_576_324_videos_for_testing_5frames

from vmaf.core.quality_runner import SpeedTemporalQualityRunner
from vmaf.tools.misc import MyTestCase


class SpeedTemporalQualityRunnerTest(MyTestCase):

    def test_run_speed_temporal_runner(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.qrunner = SpeedTemporalQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.qrunner.run(parallelize=False)
        results = self.qrunner.results

        self.assertAlmostEqual(results[0]['SpeedTemporal_score'], 18.9411808, places=4)
        self.assertAlmostEqual(results[1]['SpeedTemporal_score'], 0.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
