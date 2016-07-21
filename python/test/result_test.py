__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
from functools import partial

import numpy as np

from core.asset import Asset
import config
from core.result import Result
from core.result_store import FileSystemResultStore
from core.quality_runner import VmafLegacyQualityRunner
from tools.stats import ListStats

class ResultTest(unittest.TestCase):

    def setUp(self):
        ref_path = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv"
        dis_path = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_1_0.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafLegacyQualityRunner(
            [asset], None, fifo_mode=True,
            delete_workdir=True, result_store=FileSystemResultStore(),
        )
        self.runner.run()
        self.result = self.runner.results[0]

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def test_todataframe_fromdataframe(self):

        print 'test on result to/from dataframe...'
        df = self.result.to_dataframe()
        df_vmaf = df.loc[df['scores_key'] == 'VMAF_legacy_scores']
        df_adm = df.loc[df['scores_key'] == 'VMAF_feature_adm_scores']
        df_vif = df.loc[df['scores_key'] == 'VMAF_feature_vif_scores']
        df_ansnr = df.loc[df['scores_key'] == 'VMAF_feature_ansnr_scores']
        df_motion = df.loc[df['scores_key'] == 'VMAF_feature_motion_scores']
        df_adm_den = df.loc[df['scores_key'] == 'VMAF_feature_adm_den_scores']
        self.assertEquals(len(df), 24)
        self.assertEquals(len(df_vmaf), 1)
        self.assertEquals(len(df_adm), 1)
        self.assertEquals(len(df_vif), 1)
        self.assertEquals(len(df_ansnr), 1)
        self.assertEquals(len(df_motion), 1)
        self.assertAlmostEquals(np.mean(df_vmaf.iloc[0]['scores']), 44.4942308947, places=4)
        self.assertAlmostEquals(np.mean(df_adm.iloc[0]['scores']), 0.813856666667, places=4)
        self.assertAlmostEquals(np.mean(df_vif.iloc[0]['scores']), 0.156834666667, places=4)
        self.assertAlmostEquals(np.mean(df_ansnr.iloc[0]['scores']), 7.92623066667, places=4)
        self.assertAlmostEquals(np.mean(df_motion.iloc[0]['scores']), 12.5548366667, places=4)
        self.assertAlmostEquals(np.mean(df_adm_den.iloc[0]['scores']), 30814.9100813, places=3)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_legacy_scores', 'scores')), 44.4942308947, places=4)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_feature_adm_scores', 'scores')), 0.813856666667, places=4)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_feature_vif_scores', 'scores')), 0.156834666667, places=4)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_feature_ansnr_scores', 'scores')), 7.92623066667, places=4)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_feature_motion_scores', 'scores')), 12.5548366667, places=4)
        self.assertEquals(df.iloc[0]['dataset'], 'test')
        self.assertEquals(df.iloc[0]['content_id'], 0)
        self.assertEquals(df.iloc[0]['asset_id'], 0)
        self.assertEquals(df.iloc[0]['ref_name'], 'checkerboard_1920_1080_10_3_0_0.yuv')
        self.assertEquals(df.iloc[0]['dis_name'], 'checkerboard_1920_1080_10_3_1_0.yuv')
        self.assertEquals(
            df.iloc[0]['asset'],
            '{"asset_dict": {"height": 1080, "use_path_as_workpath": 1, "width": 1920}, "asset_id": 0, "content_id": 0, "dataset": "test", "dis_path": "checkerboard_1920_1080_10_3_1_0.yuv", "ref_path": "checkerboard_1920_1080_10_3_0_0.yuv", "workdir": ""}')
        self.assertEquals(df.iloc[0]['executor_id'], 'VMAF_legacy_V1.2')

        Result._assert_asset_dataframe(df)

        recon_result = Result.from_dataframe(df)
        self.assertEquals(self.result, recon_result)
        self.assertTrue(self.result == recon_result)
        self.assertFalse(self.result != recon_result)

    def test_to_score_str(self):
        print 'test on result aggregate scores...'
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 44.494230894688833 , places=4)
        self.assertAlmostEquals(self.result['VMAF_legacy_score'], 44.494230894688833, places=4)
        self.assertAlmostEquals(self.result.get_score('VMAF_feature_adm_score'), 0.81386, places=4)
        self.assertAlmostEquals(self.result['VMAF_feature_adm_score'], 0.81386, places=4)
        self.assertAlmostEquals(self.result['VMAF_feature_vif_score'], 0.15683466666666665 , places=4)
        self.assertAlmostEquals(self.result['VMAF_feature_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEquals(self.result['VMAF_feature_ansnr_score'], 7.92623066667, places=4)
        self.result.set_aggregate_method(np.min)
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 42.084764558485965, places=4)
        self.result.set_aggregate_method(np.max)
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 48.701976416017928, places=4)
        self.result.set_aggregate_method(np.median)
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 42.695951709562614, places=4)
        self.result.set_aggregate_method(np.mean)
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 44.494230894688833, places=4)
        self.result.set_aggregate_method(np.std)
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 2.9857694946316129, places=4)
        self.result.set_aggregate_method(np.var)
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 8.9148194750727168, places=4)
        self.result.set_aggregate_method(partial(np.percentile, q=50))
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 42.695951709562614, places=4)
        self.result.set_aggregate_method(partial(np.percentile, q=80))
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 46.299566533435808, places=4)
        self.result.set_aggregate_method(ListStats.total_variation)
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 6.3116182819936384, places=4)
        self.result.set_aggregate_method(partial(ListStats.moving_average, n=2))
        self.assertItemsEqual(self.result.get_score('VMAF_legacy_score'),
                              [46.922334053546891,  46.922334053546891,  46.922334053546891])

        with self.assertRaises(KeyError):
            self.result.get_score('VVMAF_legacy_score')
        with self.assertRaises(KeyError):
            self.result.get_score('VMAF_motion_scor')

class ResultStoreTest(unittest.TestCase):

    def setUp(self):
        ref_path = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv"
        dis_path = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_1_0.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafLegacyQualityRunner(
            [asset], None, fifo_mode=True,
            delete_workdir=True, result_store=None,
        )
        self.runner.run()
        self.result = self.runner.results[0]

    def tearDown(self):
        if hasattr(self, 'result') and hasattr(self, 'result_store'):
            self.result_store.delete(self.result.asset, self.result.executor_id)
        pass

    def test_file_system_result_store_save_load(self):
        print 'test on file system result store save and load...'
        self.result_store = FileSystemResultStore(logger=None)
        asset = self.result.asset
        executor_id = self.result.executor_id

        self.result_store.save(self.result)

        loaded_result = self.result_store.load(asset, executor_id)

        self.assertEquals(self.result, loaded_result)


if __name__ == '__main__':
    unittest.main()
