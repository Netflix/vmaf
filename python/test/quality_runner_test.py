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
    EnsembleVmafQualityRunner, VmafnegQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from vmaf.tools.misc import MyTestCase
from vmaf.tools.stats import ListStats

from test.testutil import set_default_576_324_videos_for_testing, set_default_flat_1920_1080_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing, set_default_576_324_12bit_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class QualityRunnerTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
        super().tearDown()

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
        self.runner.run(parallelize=True)

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

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing()

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)

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

    def test_run_vmaf_legacy_runner_12le(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.5129766666666666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 2.932176666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 24.906395333333336, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 2.932176666666667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.004588333333334, places=4)

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 72.18465772375357, places=4)
        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 95.94112242732263, places=4)

    def test_run_vmaf_legacy_runner_with_result_store(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        result_store = FileSystemResultStore(logger=None)

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=result_store
        )

        self.runner.run(parallelize=True)
        result0, result1 = self.runner.results

        # NOTE: since stored results are actually VMAF_feature's not VMAF's,
        # the two paths below shouldn't exist
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result1)))

        self.runner.run(parallelize=True)
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
                'model_filepath': VmafConfig.model_path("other_models", "nflx_v1.json"),
            },
            optional_dict2=None,
        )
        self.runner.run(parallelize=True)

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
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_ansnr_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_3threads(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict2={'n_threads': 3}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_v061(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1.json"),
            }
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
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 92.52240518580402, places=4)
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
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_Phone_score'], 92.52240518580402, places=4)
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
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.053996580527295335, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 0.999998395234, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 1.00000122625, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 0.999998263056, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 1.0000000801, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[2]['VMAF_integer_feature_vif_scale0_score'], 0.112931470868, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_vif_scale1_score'], 0.2983721613615637, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_vif_scale2_score'], 0.33743204896754614, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_vif_scale3_score'], 0.496419716304, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_motion2_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_adm2_score'], 0.7853384465157921, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 7.985898744818505, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.14289994394608, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 35.06866714286451, places=4)

    def test_run_vmaf_runner_flat(self):

        ref_path, dis_path, asset, asset_original = set_default_flat_1920_1080_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 97.42804264261031, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 97.428042675471147, places=4)

    def test_run_vmaf_runner_with_norm_type_none(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={
                'model_filepath': VmafConfig.model_path("other_models", "nflxtrain_norm_type_none.json"),
            },
            optional_dict2=None,
        )
        self.runner.run(parallelize=True)

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
        self.runner.run(parallelize=True)

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
                'model_filepath': [VmafConfig.model_path("vmaf_float_v0.6.1.json"), VmafConfig.model_path("other_models", "vmaf_v0.6.0.json")],
            },
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)

        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_0_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_1_score'], 81.77005183877434, places=3)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_score'], 79.22715378972225, places=4)

    def test_run_psnr_runner(self):

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

        self.assertAlmostEqual(results[0]['PSNR_scores'][2], 30.993823, places=4)

    def test_run_ssim_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = SsimQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
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

    def test_run_ms_ssim_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = MsSsimQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
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

    def test_run_vmaf_runner_pool_harmonic_mean(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        for result in results:
            result.set_score_aggregate_method(ListStats.harmonic_mean)

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.51000590218766, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.94504634354891, places=4)

    def test_run_vmaf_runner_pool_perc10(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        for result in results:
            result.set_score_aggregate_method(ListStats.perc10)

        self.assertAlmostEqual(results[0]['VMAF_score'], 72.71845922683059, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_adm2_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = Adm2QualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

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
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VIF_score'], 0.44642331250000006, places=4)
        self.assertAlmostEqual(results[1]['VIF_score'], 1.0, places=4)

    def test_run_vif2_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = Vif2QualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VIF2_score'], 0.72749630372849, places=4)
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
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40.json"),
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

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
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_outltein.json"),
                'enable_transform_score': True,
                'disable_clip_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

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

    def test_run_vmaf_runner_with_transform_score3(self):

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
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_piecewiselinear.json"),
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

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

        self.assertAlmostEqual(results[0]['VMAF_score'], 8.262602639723815, places=4)

    def test_run_vmaf_runner_with_transform_score4(self):

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
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_piecewiselinear_forced.json"),
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

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

        self.assertAlmostEqual(results[0]['VMAF_score'], 8.262602639723815, places=4)

    def test_run_vmaf_runner_with_transform_score_both_specified(self):

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
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_piecewiselinear_forced.json"),
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 8.262602639723815, places=4)

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
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40.json"),
                'enable_transform_score': False,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

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
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.053996580527295335, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 14.98274738501623, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
