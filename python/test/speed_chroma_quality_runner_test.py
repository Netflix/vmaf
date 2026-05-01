import unittest
from test.testutil import set_default_576_324_videos_for_testing_5frames

from vmaf.core.quality_runner import SpeedChromaUQualityRunner, SpeedChromaVQualityRunner, \
    SpeedChromaQualityRunner
from vmaf.tools.misc import MyTestCase


class SpeedChromaQualityRunnerTest(MyTestCase):

    def test_run_speed_chroma_runner(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.qrunner = SpeedChromaQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.qrunner.run(parallelize=False)
        results = self.qrunner.results

        self.assertAlmostEqual(results[0]['SpeedChroma_score'], 7.8327238, places=4)
        self.assertAlmostEqual(results[1]['SpeedChroma_score'], 0.0, places=4)

    def test_run_speed_chroma_u_runner(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.qrunner = SpeedChromaUQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.qrunner.run(parallelize=False)
        results = self.qrunner.results

        self.assertAlmostEqual(results[0]['SpeedChromaU_score'], 11.7757982, places=4)
        self.assertAlmostEqual(results[1]['SpeedChromaU_score'], 0.0, places=4)

    def test_run_speed_chroma_v_runner(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing_5frames()
        self.qrunner = SpeedChromaVQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.qrunner.run(parallelize=False)
        results = self.qrunner.results

        self.assertAlmostEqual(results[0]['SpeedChromaV_score'], 3.8896207999999994, places=4)
        self.assertAlmostEqual(results[1]['SpeedChromaV_score'], 0.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
