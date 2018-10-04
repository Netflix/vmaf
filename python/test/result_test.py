from testutil import set_default_576_324_videos_for_testing

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import json
import unittest
from functools import partial

import numpy as np

from vmaf.core.asset import Asset
from vmaf.config import VmafConfig
from vmaf.core.result import Result
from vmaf.core.result_store import FileSystemResultStore
from vmaf.core.quality_runner import VmafLegacyQualityRunner, VmafQualityRunner
from vmaf.tools.stats import ListStats

class ResultTest(unittest.TestCase):

    def setUp(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_1_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
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

    def test_todataframe_fromdataframe(self):

        print 'test on result to/from dataframe...'
        df = self.result.to_dataframe()
        df_vmaf = df.loc[df['scores_key'] == 'VMAF_legacy_scores']
        df_adm = df.loc[df['scores_key'] == 'VMAF_feature_adm_scores']
        df_vif = df.loc[df['scores_key'] == 'VMAF_feature_vif_scores']
        df_ansnr = df.loc[df['scores_key'] == 'VMAF_feature_ansnr_scores']
        df_motion = df.loc[df['scores_key'] == 'VMAF_feature_motion_scores']
        df_adm_den = df.loc[df['scores_key'] == 'VMAF_feature_adm_den_scores']
        self.assertEquals(len(df), 38)
        self.assertEquals(len(df_vmaf), 1)
        self.assertEquals(len(df_adm), 1)
        self.assertEquals(len(df_vif), 1)
        self.assertEquals(len(df_ansnr), 1)
        self.assertEquals(len(df_motion), 1)
        self.assertAlmostEquals(np.mean(df_vmaf.iloc[0]['scores']), 40.421899030550769, places=4)
        self.assertAlmostEquals(np.mean(df_adm.iloc[0]['scores']), 0.78533833333333336, places=4)
        self.assertAlmostEquals(np.mean(df_vif.iloc[0]['scores']), 0.156834666667, places=4)
        self.assertAlmostEquals(np.mean(df_ansnr.iloc[0]['scores']), 7.92623066667, places=4)
        self.assertAlmostEquals(np.mean(df_motion.iloc[0]['scores']), 12.5548366667, places=4)
        self.assertAlmostEquals(np.mean(df_adm_den.iloc[0]['scores']), 2773.8912249999998, places=3)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_legacy_scores', 'scores')), 40.421899030550769, places=4)
        self.assertAlmostEquals(np.mean(Result.get_unique_from_dataframe(df, 'VMAF_feature_adm_scores', 'scores')), 0.78533833333333336, places=4)
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
            '{"asset_dict": {"height": 1080, "use_path_as_workpath": 1, "width": 1920}, "asset_id": 0, "content_id": 0, "dataset": "test", "dis_path": "checkerboard_1920_1080_10_3_1_0.yuv", "ref_path": "checkerboard_1920_1080_10_3_0_0.yuv", "workdir": ""}') # noqa
        self.assertEquals(df.iloc[0]['executor_id'], 'VMAF_legacy_VF0.2.4c-1.1')

        Result._assert_asset_dataframe(df)

        recon_result = Result.from_dataframe(df)
        self.assertEquals(self.result, recon_result)
        self.assertTrue(self.result == recon_result)
        self.assertFalse(self.result != recon_result)

    def test_to_score_str(self):
        print 'test on result aggregate scores...'
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 40.421899030550769, places=4)
        self.assertAlmostEquals(self.result['VMAF_legacy_score'], 40.421899030550769, places=4)
        self.assertAlmostEquals(self.result.get_result('VMAF_feature_adm_score'), 0.78533833333333336, places=4)
        self.assertAlmostEquals(self.result['VMAF_feature_adm_score'], 0.78533833333333336, places=4)
        self.assertAlmostEquals(self.result['VMAF_feature_vif_score'], 0.15683466666666665, places=4)
        self.assertAlmostEquals(self.result['VMAF_feature_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEquals(self.result['VMAF_feature_ansnr_score'], 7.92623066667, places=4)
        self.result.set_score_aggregate_method(np.min)
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 37.573531379639725, places=4)
        self.result.set_score_aggregate_method(np.max)
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 44.815357234059327, places=4)
        self.result.set_score_aggregate_method(np.median)
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 38.876808477953254, places=4)
        self.result.set_score_aggregate_method(np.mean)
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 40.421899030550769, places=4)
        self.result.set_score_aggregate_method(np.std)
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 3.1518765879212993, places=4)
        self.result.set_score_aggregate_method(np.var)
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 9.9343260254864134, places=4)
        self.result.set_score_aggregate_method(partial(np.percentile, q=50))
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 38.876808477953254, places=4)
        self.result.set_score_aggregate_method(partial(np.percentile, q=80))
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 42.439937731616901, places=4)
        self.result.set_score_aggregate_method(ListStats.total_variation)
        self.assertAlmostEquals(self.result.get_result('VMAF_legacy_score'), 6.5901873052628375, places=4)
        self.result.set_score_aggregate_method(partial(ListStats.moving_average, n=2))
        self.assertItemsEqual(self.result.get_result('VMAF_legacy_score'),
                              [42.86773029545774, 42.86773029545774, 42.86773029545774])

        with self.assertRaises(KeyError):
            self.result.get_result('VVMAF_legacy_score')
        with self.assertRaises(KeyError):
            self.result.get_result('VMAF_motion_scor')

class ResultFormattingTest(unittest.TestCase):

    def setUp(self):

        # ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        # dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_1_0.yuv")
        # asset = Asset(dataset="test", content_id=0, asset_id=0,
        #               workdir_root=VmafConfig.workdir_path(),
        #               ref_path=ref_path,
        #               dis_path=dis_path,
        #               asset_dict={'width':1920, 'height':1080})
        #
        # self.runner = SsimQualityRunner(
        #     [asset], None, fifo_mode=True,
        #     delete_workdir=True, result_store=FileSystemResultStore(),
        # )
        # self.runner.run()
        #
        # FileSystemResultStore.save_result(self.runner.results[0], VmafConfig.test_resource_path('/ssim_result_for_test.txt')

        self.result = FileSystemResultStore.load_result(VmafConfig.test_resource_path('ssim_result_for_test.txt'))

    def tearDown(self):
        # if hasattr(self, 'runner'):
        #     self.runner.remove_results()
        pass

    @unittest.skip("numerical value has changed.")
    def test_to_xml(self):
        self.assertEquals(self.result.to_xml().strip(), """
<?xml version="1.0" ?>
<result executorId="SSIM_V1.0">
  <asset identifier="test_0_0_checkerboard_1920_1080_10_3_0_0_1920x1080_vs_checkerboard_1920_1080_10_3_1_0_1920x1080_q_1920x1080"/>
  <frames>
    <frame SSIM_feature_ssim_c_score="0.997404" SSIM_feature_ssim_l_score="0.999983" SSIM_feature_ssim_s_score="0.935802" SSIM_score="0.933353" frameNum="0"/>
    <frame SSIM_feature_ssim_c_score="0.997404" SSIM_feature_ssim_l_score="0.999983" SSIM_feature_ssim_s_score="0.935802" SSIM_score="0.933353" frameNum="1"/>
    <frame SSIM_feature_ssim_c_score="0.997404" SSIM_feature_ssim_l_score="0.999983" SSIM_feature_ssim_s_score="0.935803" SSIM_score="0.933354" frameNum="2"/>
  </frames>
  <aggregate SSIM_feature_ssim_c_score="0.997404" SSIM_feature_ssim_l_score="0.999983" SSIM_feature_ssim_s_score="0.935802333333" SSIM_score="0.933353333333" method="mean"/>
</result>
        """.strip())

    def test_to_json(self):
        self.assertEquals(self.result.to_dict(), json.loads("""
{
    "executorId": "SSIM_V1.0",
    "asset": {
        "identifier": "test_0_0_checkerboard_1920_1080_10_3_0_0_1920x1080_vs_checkerboard_1920_1080_10_3_1_0_1920x1080_q_1920x1080"
    },
    "frames": [
        {
            "frameNum": 0,
            "SSIM_feature_ssim_c_score": 0.997404,
            "SSIM_feature_ssim_l_score": 0.999983,
            "SSIM_feature_ssim_s_score": 0.935802,
            "SSIM_score": 0.933353
        },
        {
            "frameNum": 1,
            "SSIM_feature_ssim_c_score": 0.997404,
            "SSIM_feature_ssim_l_score": 0.999983,
            "SSIM_feature_ssim_s_score": 0.935802,
            "SSIM_score": 0.933353
        },
        {
            "frameNum": 2,
            "SSIM_feature_ssim_c_score": 0.997404,
            "SSIM_feature_ssim_l_score": 0.999983,
            "SSIM_feature_ssim_s_score": 0.935803,
            "SSIM_score": 0.933354
        }
    ],
    "aggregate": {
        "SSIM_feature_ssim_c_score": 0.99740399999999996,
        "SSIM_feature_ssim_l_score": 0.99998299999999996,
        "SSIM_feature_ssim_s_score": 0.93580233333333329,
        "SSIM_score": 0.93335333333333337,
        "method": "mean"
    }
}
        """))


class ResultStoreTest(unittest.TestCase):

    def setUp(self):
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_1_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
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

class ResultStoreTestWithNone(unittest.TestCase):

    def test_load_result_with_none(self):
        print 'test on file system result store load result with None...'
        result = FileSystemResultStore.load_result(VmafConfig.test_resource_path('result_with_none.txt'))
        result.set_score_aggregate_method(ListStats.nonemean)
        self.assertAlmostEqual(result['STRRED_feature_srred_score'], 5829.2644469999996, places=4)

class ResultAggregatingTest(unittest.TestCase):

    def test_from_xml_from_json_and_aggregation(self):

        print 'test on running from_xml and from_json and aggregation...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        asset_list = [asset, asset_original]

        self.runner = VmafQualityRunner(
            asset_list,
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_v0.6.1.pkl"),
            },
            optional_dict2=None,
        )
        self.runner.run()

        results = self.runner.results

        xml_string_expected = results[0].to_xml()
        xml_string_recon = Result.from_xml(xml_string_expected).to_xml()

        json_string_expected = results[0].to_json()
        json_string_recon = Result.from_json(json_string_expected).to_json()

        assert xml_string_expected == xml_string_recon, "XML files do not match"
        assert json_string_expected == json_string_recon, "JSON files do not match"

        combined_result = Result.combine_result([results[0], results[1]])

        # check that all keys are there
        combined_result_keys = [key for key in combined_result.result_dict]
        keys_0 = [key for key in results[0].result_dict]
        keys_1 = [key for key in results[1].result_dict]
        assert set(keys_0) == set(keys_1) == set(combined_result_keys)

        # check that the dictionaries have been copied as expected
        for key in combined_result_keys:
            assert len(combined_result.result_dict[key]) == len(results[0].result_dict[key]) + len(results[1].result_dict[key])
            assert combined_result.result_dict[key][0] == results[0].result_dict[key][0]
            assert combined_result.result_dict[key][len(results[0].result_dict[key]) - 1] == results[0].result_dict[key][len(results[0].result_dict[key]) - 1]
            assert combined_result.result_dict[key][len(results[0].result_dict[key])] == results[1].result_dict[key][0]
            assert combined_result.result_dict[key][len(combined_result.result_dict[key]) - 1] == results[1].result_dict[key][len(results[1].result_dict[key]) - 1]

if __name__ == '__main__':
    unittest.main()
