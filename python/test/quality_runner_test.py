from __future__ import absolute_import

import os
import sys
import unittest

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset, NorefAsset
from vmaf.core.quality_runner import VmafLegacyQualityRunner, VmafQualityRunner, \
    PsnrQualityRunner, MsSsimQualityRunner, \
    SsimQualityRunner, Adm2QualityRunner, VmafPhoneQualityRunner, VifQualityRunner, \
    Vif2QualityRunner, BootstrapVmafQualityRunner, BaggingVmafQualityRunner, NiqeQualityRunner, \
    EnsembleVmafQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from vmaf.tools.stats import ListStats

from test.testutil import set_default_576_324_videos_for_testing, set_default_flat_1920_1080_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class QualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480})
        runner = VmafLegacyQualityRunner([asset], None)
        self.assertEqual(runner.executor_id, 'VMAF_legacy_VF0.2.7-1.1')

    def test_run_vmaf_legacy_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'],0.4460930625000001, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833337, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 65.37503585467225, places=4)
        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 96.444658329804156, places=4)

    def test_run_vmaf_legacy_runner_10le(self):

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv422p10le.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv422p10le.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'yuv_type': 'yuv422p10le'})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'yuv_type': 'yuv422p10le'})

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833333, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 65.37503585467225, places=4)
        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 96.444658329804156, places=4)

    def test_run_vmaf_legacy_runner_with_result_store(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        result_store = FileSystemResultStore(logger=None)

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=result_store
        )

        self.runner.run()
        result0, result1 = self.runner.results

        # NOTE: since stored results are actually VMAF_feature's not VMAF's,
        # the two paths below shouldn't exist
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result1)))

        self.runner.run()
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833333, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 65.37503585467225, places=4)
        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 96.444658329804156, places=4)

    def test_run_vmaf_runner_v1_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("other_models", "nflx_v1.pkl"),
            },
            optional_dict2=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 77.17414738991636, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_vmaf_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_with_phone_score(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'enable_transform_score': True,
            }
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 92.53270050911406, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_vmaf_phone_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        with self.assertRaises(AssertionError):
            VmafPhoneQualityRunner(
                [asset, asset_original],
                None, fifo_mode=True,
                delete_workdir=True,
                result_store=None,
                optional_dict={
                    'enable_transform_score': True,
                }
            )

        self.runner = VmafPhoneQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_Phone_score'], 92.53270050911406, places=4)
        self.assertAlmostEqual(results[1]['VMAF_Phone_score'], 100.0, places=4)

    def test_run_vmaf_runner_checkerboard(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        dis_path2 = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_1_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 1920, 'height': 1080})

        asset2 = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path2,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafQualityRunner(
            [asset, asset_original, asset2],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.053996580527295335, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 0.999998395234, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.00000122625, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999998263056, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0000000801, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[2]['VMAF_feature_vif_scale0_score'], 0.112931470868, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_vif_scale1_score'], 0.298448603112, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_vif_scale2_score'], 0.337612207676, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_vif_scale3_score'], 0.496419716304, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_adm2_score'], 0.7853384465157921, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 7.985956215118768, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.142659046424384, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 35.066157497128764, places=4)

    def test_run_vmaf_runner_flat(self):

        ref_path, dis_path, asset, asset_original = set_default_flat_1920_1080_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 97.427927701008869, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 97.428042675471147, places=4)

    @unittest.skipIf(sys.version_info < (3,), reason="For py3 only: model not supported by py2.")
    def test_run_vmaf_runner_with_rf_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={
                'model_filepath': VmafConfig.model_path("other_models", "nflx_vmaff_rf_v2.pkl"),
            },
            optional_dict2=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'],4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 79.21274038461537, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_vmaf_runner_with_norm_type_none(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={
                'model_filepath': VmafConfig.model_path("other_models", "nflxtrain_norm_type_none.pkl"),
            },
            optional_dict2=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 74.25323556658451, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 77.996338095161946, places=4)

    def test_run_ensemblevmaf_runner_same_models(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = EnsembleVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)

        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_0_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_1_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_score'], 76.68425574067017, places=4)

    def test_run_ensemblevmaf_runner_different_models(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = EnsembleVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': [VmafConfig.model_path("vmaf_v0.6.1.pkl"), VmafConfig.model_path("other_models", "vmaf_v0.6.0.pkl")],
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)

        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_0_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_1_score'], 81.77005183877434, places=4)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_score'], 79.22715378972225, places=4)

    def test_run_psnr_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = PsnrQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)

        self.assertAlmostEqual(results[0]['PSNR_scores'][2], 30.993823, places=4)

    def test_run_ssim_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = SsimQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['SSIM_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_l_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_c_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_s_score'], 0.89773633333333336, places=4)

        self.assertAlmostEqual(results[1]['SSIM_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_l_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_c_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_s_score'], 1.0, places=4)

    def test_run_ms_ssim_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = MsSsimQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['MS_SSIM_score'], 0.9632498125, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale0_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale0_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale0_score'], 0.89773633333333336, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale1_score'], 0.99899612500000001, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale1_score'], 0.9857694375, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale1_score'], 0.941185875, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale2_score'], 0.99923564583333324, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale2_score'], 0.997034020833, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale2_score'], 0.977992145833, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale3_score'], 0.99929210416666658, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale3_score'], 0.999588104167, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale3_score'], 0.99387125, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale4_score'], 0.99940356249999995, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale4_score'], 0.999907625, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale4_score'], 0.998222583333, places=4)

        self.assertAlmostEqual(results[1]['MS_SSIM_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale4_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale4_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale4_score'], 1., places=4)

    def test_run_vmaf_runner_pool_harmonic_mean(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        for result in results:
            result.set_score_aggregate_method(ListStats.harmonic_mean)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.36259426848527943, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.76614783252704499, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.86261788202730005, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.91587319935214206, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.5480006115798028, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9344716687697119, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.5480006115798028, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.52607825893074, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.94504634354891, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 1.0, places=4)

    def test_run_vmaf_runner_pool_perc10(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        for result in results:
            result.set_score_aggregate_method(ListStats.perc10)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3324451679468679, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.73826183605283979, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.84227039935569437, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9023292478206113, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.3064741, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9237900248130357, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.3064741, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 72.7295739803547, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_adm2_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = Adm2QualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['ADM2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[1]['ADM2_score'], 1.0, places=4)

    def test_run_vif_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VifQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VIF_score'], 0.44609339583333335, places=4)
        self.assertAlmostEqual(results[1]['VIF_score'], 1.0, places=4)

    def test_run_vif2_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = Vif2QualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VIF2_score'], 0.7272233644583975, places=4)
        self.assertAlmostEqual(results[1]['VIF2_score'], 1.0, places=4)

    def test_run_vmaf_runner_with_transform_score(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40.pkl"),
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 32.757433750978919, places=4)

    def test_run_vmaf_runner_with_transform_score_2(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_outltein.pkl"),
                'enable_transform_score': True,
                'disable_clip_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], -7.2425662490210838, places=4)

    def test_run_vmaf_runner_with_transform_score_disabled(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40.pkl"),
                'enable_transform_score': False,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 0.0, places=4)

    def test_run_vmaf_runner_with_transform_for_phone(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.053996580527295335, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 14.982837886251568, places=4)

    def test_run_bootstrap_vmaf_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.pkl'),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 75.42800743529182, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 99.95804893252175, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 75.11496350766362, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 99.96504855577571, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 0.6816714405308405, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 0.03947607207290399, places=4)

    def test_run_bootstrap_vmaf_runner_with_transform_score(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.pkl'),
                'enable_transform_score': True,
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 91.71304091659904, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 91.50418140067052, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 0.4609151507341837, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 0.0, places=10)

    def test_run_bootstrap_vmaf_runner_specific_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path('vmaf_rb_v0.6.2', 'vmaf_rb_v0.6.2.pkl'),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 75.42800743529182, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 99.95804893252175, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 73.08851325162333, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 99.686116179979152, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 1.2292399440451214, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 1.5917514683608882, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_ci95_low_score'], 70.79156057918671, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_ci95_low_score'], 94.784491176494996, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_ci95_high_score'], 74.84074207742032, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_ci95_high_score'], 99.992560767034618, places=4)

    def test_run_bootstrap_vmaf_runner_residue_bootstrap_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path('vmaf_rb_v0.6.3', 'vmaf_rb_v0.6.3.pkl'),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 75.42800743529182, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 99.95804893252175, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 73.09131553704874, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 99.79000465995409, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 1.1982762081883995, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 1.3028824838324222, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_ci95_low_score'], 70.81472328674501, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_ci95_low_score'], 94.79667446930989, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_ci95_high_score'], 74.83768715705374, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_ci95_high_score'], 99.99736657892976, places=4)

    def test_run_bootstrap_vmaf_runner_default_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 75.42800743529182, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 99.95804893252175, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 74.94883517626657, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 99.93908291255723, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 1.3121299437588092, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 0.09930398700617331, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_ci95_low_score'], 72.97208643408686, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_ci95_low_score'], 91.1152017392609, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_ci95_high_score'], 77.36908154045717, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_ci95_high_score'], 100.0, places=4)

    def test_run_bootstrap_vmaf_runner_default_model_with_transform_score(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'enable_transform_score': True,
            }
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 91.71304091659904, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 91.39500100024217, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 0.8770874088948243, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 0.0, places=4)

    def test_run_bootstrap_vmaf_runner_10models(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test2.pkl'),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 75.42800743529182, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 99.95804893252175, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 75.1541667848584, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 99.9640738745435, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 1.415440563303708, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 0.03321535846597722, places=4)

    def test_run_bagging_vmaf_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BaggingVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.pkl'),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['BAGGING_VMAF_score'], 75.11496350766362, places=4)
        self.assertAlmostEqual(results[1]['BAGGING_VMAF_score'], 99.965048555775709, places=4)
        self.assertAlmostEqual(results[0]['BAGGING_VMAF_bagging_score'], 75.11496350766362, places=4)
        self.assertAlmostEqual(results[1]['BAGGING_VMAF_bagging_score'], 99.965048555775709, places=4)
        self.assertAlmostEqual(results[0]['BAGGING_VMAF_stddev_score'], 0.6816714405308405, places=4)
        self.assertAlmostEqual(results[1]['BAGGING_VMAF_stddev_score'], 0.03947607207290399, places=4)

    @unittest.skipIf(sys.version_info < (3,), reason="For py3 only: model not supported by py2.")
    def test_run_niqe_runner(self):

        ref1_path = VmafConfig.test_resource_path("test_image_yuv", "100007.yuv")
        ref2_path = VmafConfig.test_resource_path("test_image_yuv", "100039.yuv")
        asset1 = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      dis_path=ref1_path,
                      asset_dict={'width': 481, 'height': 321, 'yuv_type': 'yuv444p'})

        asset2 = NorefAsset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      dis_path=ref2_path,
                      asset_dict={'width': 481, 'height': 321, 'yuv_type': 'yuv444p'})

        self.runner = NiqeQualityRunner(
            [asset1, asset2],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['NIQE_score'], 4.8656072348129422, places=4)
        self.assertAlmostEqual(results[1]['NIQE_score'], 2.9309929860778756, places=2)

    def test_run_vmaf_runner_with_4k_1d5H_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_4k_v0.6.1.pkl"),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 84.97241763181641, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_bootstrap_vmaf_runner_with_4k_1d5H_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path('vmaf_4k_rb_v0.6.2', 'vmaf_4k_rb_v0.6.2.pkl'),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 84.97241763181641, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 83.97551251514652, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 99.976138971781452, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 0.9285768709567926, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 0.03627354379389184, places=4)

    def test_run_vmaf_runner_with_bootstrap_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.pkl'),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 75.42800743529182, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.95804893252175, places=4)

    def test_run_vmaf_runner_input160x90(self):

        ref_path = VmafConfig.test_resource_path("yuv", "ref_test_0_1_src01_hrc00_576x324_576x324_vs_src01_hrc01_576x324_576x324_q_160x90.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "dis_test_0_1_src01_hrc00_576x324_576x324_vs_src01_hrc01_576x324_576x324_q_160x90.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 160, 'height': 90})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                               workdir_root=VmafConfig.workdir_path(),
                               ref_path=ref_path,
                               dis_path=ref_path,
                               asset_dict={'width': 160, 'height': 90})

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.6890852611788579, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.9568544348982971, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.97696930627404, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9830559447224111, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 1.3596959375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9807496975664337, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 1.3596959375, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 92.53369746427056, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.3088521349021, places=4)


class ParallelQualityRunnerTest(unittest.TestCase):

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runners'):
            for runner in self.runners:
                runner.remove_results()
            pass

    def test_run_parallel_vmaf_legacy_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625000001, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833337, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 65.37503585467225, places=4)
        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 96.444658329804156, places=4)
        self.assertAlmostEqual(results[2]['VMAF_legacy_score'], 96.444658329804156, places=4)

    def test_run_parallel_vmaf_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original, asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_parallel_psnr_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = PsnrQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)

    def test_run_parallel_psnr_runner_processes(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = PsnrQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True, processes=1)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)

    @unittest.skipIf(sys.version_info < (3,), reason="For py3 only: model not supported by py2.")
    def test_run_parallel_vamf_runner_with_rf_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original, asset, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("other_models", "nflx_vmaff_rf_v2.pkl"),
            },
            optional_dict2=None,
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'],4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 79.21274038461537, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 79.21274038461537, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 79.21274038461537, places=4)
        self.assertAlmostEqual(results[4]['VMAF_score'], 79.21274038461537, places=4)

    def test_run_parallel_ssim_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = SsimQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['SSIM_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_l_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_c_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_s_score'], 0.89773633333333336, places=4)

        self.assertAlmostEqual(results[1]['SSIM_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_l_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_c_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_s_score'], 1.0, places=4)

        self.assertAlmostEqual(results[2]['SSIM_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[3]['SSIM_score'], 0.86322654166666657, places=4)

    def test_run_parallel_msssim_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = MsSsimQualityRunner(
            [asset, asset_original, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['MS_SSIM_score'], 0.9632498125, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale0_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale0_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale0_score'], 0.89773633333333336, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale1_score'], 0.99899612500000001, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale1_score'], 0.9857694375, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale1_score'], 0.941185875, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale2_score'], 0.99923564583333324, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale2_score'], 0.997034020833, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale2_score'], 0.977992145833, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale3_score'], 0.99929210416666658, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale3_score'], 0.999588104167, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale3_score'], 0.99387125, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale4_score'], 0.99940356249999995, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale4_score'], 0.999907625, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale4_score'], 0.998222583333, places=4)

        self.assertAlmostEqual(results[1]['MS_SSIM_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale4_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale4_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale4_score'], 1., places=4)

        self.assertAlmostEqual(results[2]['MS_SSIM_score'], 0.9632498125, places=4)

    def test_run_parallel_msssim_runner_with_result_store(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = MsSsimQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['MS_SSIM_score'], 0.9632498125, places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_score'], 1.0, places=4)
        self.assertAlmostEqual(results[2]['MS_SSIM_score'], 0.9632498125, places=4)
        self.assertAlmostEqual(results[3]['MS_SSIM_score'], 0.9632498125, places=4)

    def test_run_parallel_runner_with_repeated_assets(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = PsnrQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)
        self.assertAlmostEqual(results[2]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[3]['PSNR_score'], 30.755063979166664, places=4)

    def test_run_parallel_runner_with_parallel_disabled(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = PsnrQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=False)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)
        self.assertAlmostEqual(results[2]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[3]['PSNR_score'], 30.755063979166664, places=4)

    @unittest.skipIf(sys.version_info < (3,), reason="For py3 only: model not supported by py2.")
    def test_run_parallel_niqe_runner(self):

        ref1_path = VmafConfig.test_resource_path("test_image_yuv", "100007.yuv")
        ref2_path = VmafConfig.test_resource_path("test_image_yuv", "100039.yuv")
        asset1 = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      dis_path=ref1_path,
                      asset_dict={'width': 481, 'height': 321, 'yuv_type': 'yuv444p'})

        asset2 = NorefAsset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      dis_path=ref2_path,
                      asset_dict={'width': 481, 'height': 321, 'yuv_type': 'yuv444p'})

        self.runner = NiqeQualityRunner(
            [asset1, asset2],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={},
            optional_dict2={},
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['NIQE_score'], 4.8656072348129422, places=4)
        self.assertAlmostEqual(results[1]['NIQE_score'], 2.9309929860778756, places=2)


class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmaf_quality_runner_version(self):
        self.assertEqual(VmafQualityRunner.VERSION, 'F0.2.7-0.6.1')
        self.assertEqual(VmafQualityRunner.ALGO_VERSION, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
