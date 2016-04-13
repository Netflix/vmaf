__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

import numpy as np

from core.asset import Asset
import config
from core.result import Result
from core.result_store import FileSystemResultStore
from core.quality_runner import VmafLegacyQualityRunner


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
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            delete_workdir=True, result_store=None,
        )
        self.runner.run()
        self.result = self.runner.results[0]

    def tearDown(self):
        if hasattr(self, 'runner'): self.runner.remove_logs()
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
        self.assertEquals(len(df), 10)
        self.assertEquals(len(df_vmaf), 1)
        self.assertEquals(len(df_adm), 1)
        self.assertEquals(len(df_vif), 1)
        self.assertEquals(len(df_ansnr), 1)
        self.assertEquals(len(df_motion), 1)
        self.assertAlmostEquals(np.mean(df_vmaf.iloc[0]['scores']), 44.387054745587953)
        self.assertAlmostEquals(np.mean(df_adm.iloc[0]['scores']), 0.81971125384321342)
        self.assertAlmostEquals(np.mean(df_vif.iloc[0]['scores']), 0.15612933333333334)
        self.assertAlmostEquals(np.mean(df_ansnr.iloc[0]['scores']), 12.418291000000002)
        self.assertAlmostEquals(np.mean(df_motion.iloc[0]['scores']), 12.343795333333333)
        self.assertAlmostEquals(np.mean(df_adm_den.iloc[0]['scores']), 30814.909660333331)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_legacy_scores', 'scores')), 44.387054745587953)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_feature_adm_scores', 'scores')), 0.81971125384321342)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_feature_vif_scores', 'scores')), 0.15612933333333334)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_feature_ansnr_scores', 'scores')), 12.418291000000002)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_feature_motion_scores', 'scores')), 12.343795333333333)
        self.assertEquals(df.iloc[0]['dataset'], 'test')
        self.assertEquals(df.iloc[0]['content_id'], 0)
        self.assertEquals(df.iloc[0]['asset_id'], 0)
        self.assertEquals(df.iloc[0]['ref_name'], 'checkerboard_1920_1080_10_3_0_0.yuv')
        self.assertEquals(df.iloc[0]['dis_name'], 'checkerboard_1920_1080_10_3_1_0.yuv')
        self.assertEquals(
            df.iloc[0]['asset'],
            '{"asset_dict": {"height": 1080, "width": 1920}, "asset_id": 0, "content_id": 0, "dataset": "test", "dis_path": "checkerboard_1920_1080_10_3_1_0.yuv", "ref_path": "checkerboard_1920_1080_10_3_0_0.yuv", "workdir": ""}')
        self.assertEquals(df.iloc[0]['executor_id'], 'VMAF_legacy_V1.0')

        Result._assert_asset_dataframe(df)

        recon_result = Result.from_dataframe(df)
        self.assertEquals(self.result, recon_result)
        self.assertTrue(self.result == recon_result)
        self.assertFalse(self.result != recon_result)

    def test_to_score_str(self):
        print 'test on result aggregate scores...'
        self.assertAlmostEquals(self.result.get_score('VMAF_legacy_score'), 44.38705474558795)
        self.assertAlmostEquals(self.result.get_score('VMAF_feature_adm_score'), 0.8197112538432134)
        self.assertAlmostEquals(self.result.get_score('VMAF_feature_vif_score'), 0.15612933333333334)
        self.assertAlmostEquals(self.result.get_score('VMAF_feature_motion_score'), 12.343795333333333)
        self.assertAlmostEquals(self.result.get_score('VMAF_feature_ansnr_score'), 12.418291000000002)

        self.assertAlmostEquals(self.result.min('VMAF_legacy_scores'), 41.629151254286526)
        self.assertAlmostEquals(self.result.max('VMAF_legacy_scores'), 48.517433611970091)
        self.assertAlmostEquals(self.result.median('VMAF_legacy_scores'), 43.014579370507256)
        self.assertAlmostEquals(self.result.mean('VMAF_legacy_scores'), 44.387054745587953)
        self.assertAlmostEquals(self.result.stddev('VMAF_legacy_scores'), 2.9748809445304807)
        self.assertAlmostEquals(self.result.var('VMAF_legacy_scores'), 8.8499166341305653)
        self.assertAlmostEquals(self.result.percentile('VMAF_legacy_scores',50), 43.014579370507256)
        self.assertAlmostEquals(self.result.percentile('VMAF_legacy_scores',80), 46.316291915384959)
        self.assertAlmostEquals(self.result.total_var('VMAF_legacy_scores'), 6.1955682995731998)
        self.assertItemsEqual(self.result.moving_average('VMAF_legacy_scores',2), [46.66488916389681, 46.66488916389681, 46.66488916389681])

        with self.assertRaises(KeyError):
            self.result.min('VMAF_legacy_score')

        with self.assertRaises(KeyError):
            self.result.get_score('VVMAF_legacy_score')
        with self.assertRaises(KeyError):
            self.result.get_score('VMAF_motion_scor')

        # self.assertEquals(
        #     self.result._get_perframe_score_str(),
        #     'Frame 0: VMAF_feature_adm_score:0.799, VMAF_feature_ansnr_score:12.421, VMAF_feature_motion_score:0.000, VMAF_feature_vif_score:0.156, VMAF_legacy_score:42.112\n'
        #     'Frame 1: VMAF_feature_adm_score:0.843, VMAF_feature_ansnr_score:12.418, VMAF_feature_motion_score:18.489, VMAF_feature_vif_score:0.156, VMAF_legacy_score:47.654\n'
        #     'Frame 2: VMAF_feature_adm_score:0.800, VMAF_feature_ansnr_score:12.416, VMAF_feature_motion_score:18.542, VMAF_feature_vif_score:0.156, VMAF_legacy_score:40.617\n'
        # )
        # self.assertEquals(
        #     self.result._get_aggregate_score_str(),
        #     'Aggregate: VMAF_feature_adm_score:0.814, VMAF_feature_ansnr_score:12.418, VMAF_feature_motion_score:12.344, VMAF_feature_vif_score:0.156, VMAF_legacy_score:43.461'
        # )

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
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            delete_workdir=True, result_store=None,
        )
        self.runner.run()
        self.result = self.runner.results[0]

    def tearDown(self):
        if hasattr(self, 'runner'): self.runner.remove_logs()
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
