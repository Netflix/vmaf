import unittest
import config
from core.asset import Asset
from core.quality_runner import StrredQualityRunner
from core.result_store import FileSystemResultStore

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class ParallelQualityRunnerTestNew(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_strrred_runner(self):
        print 'test on running STRRED runner...'
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        self.runner = StrredQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['STRRED_feature_srred_score'], 4.8845008541666664, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_trred_score'], 8.9429378333333336, places=4)
        self.assertAlmostEqual(results[0]['STRRED_score'], 44.002554138184131, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_score'], 0.0, places=4)

if __name__ == '__main__':
    unittest.main()

