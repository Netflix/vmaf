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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=False)

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
        self.runner.run(parallelize=False)

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
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=result_store
        )

        self.runner.run(parallelize=False)
        result0, result1 = self.runner.results

        # NOTE: since stored results are actually VMAF_feature's not VMAF's,
        # the two paths below shouldn't exist
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result1)))

        self.runner.run(parallelize=False)
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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("other_models", "nflx_v1.json"),
            },
            optional_dict2=None,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict2={'n_threads': 3}
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_v061(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1.json"),
            }
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'enable_transform_score': True,
            }
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 92.52240518580402, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_vmaf_phone_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        with self.assertRaises(AssertionError):
            VmafPhoneQualityRunner(
                [asset, asset_original],
                None, fifo_mode=False,
                delete_workdir=True,
                result_store=None,
                optional_dict={
                    'enable_transform_score': True,
                }
            )

        self.runner = VmafPhoneQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={}
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={
                'model_filepath': VmafConfig.model_path("other_models", "nflxtrain_norm_type_none.json"),
            },
            optional_dict2=None,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)

        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_0_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_1_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_score'], 76.68425574067017, places=4)

    def test_run_ensemblevmaf_runner_different_models(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = EnsembleVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': [VmafConfig.model_path("vmaf_float_v0.6.1.json"), VmafConfig.model_path("other_models", "vmaf_v0.6.0.json")],
            },
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)

        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_0_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_model_1_score'], 81.77005183877434, places=3)
        self.assertAlmostEqual(results[0]['EnsembleVMAF_score'], 79.22715378972225, places=4)

    def test_run_psnr_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = PsnrQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=False)

        results = self.runner.results
        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)

        self.assertAlmostEqual(results[0]['PSNR_scores'][2], 30.993823, places=4)

    def test_run_ssim_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = SsimQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            result_store=None
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        for result in results:
            result.set_score_aggregate_method(ListStats.harmonic_mean)

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.51000590218766, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.94504634354891, places=4)

    def test_run_vmaf_runner_pool_perc10(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        for result in results:
            result.set_score_aggregate_method(ListStats.perc10)

        self.assertAlmostEqual(results[0]['VMAF_score'], 72.71845922683059, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_adm2_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = Adm2QualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['ADM2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[1]['ADM2_score'], 1.0, places=4)

    def test_run_vif_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VifQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VIF_score'], 0.44642331250000006, places=4)
        self.assertAlmostEqual(results[1]['VIF_score'], 1.0, places=4)

    def test_run_vif2_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = Vif2QualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40.json"),
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_outltein.json"),
                'enable_transform_score': True,
                'disable_clip_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_piecewiselinear.json"),
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_piecewiselinear_forced.json"),
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_piecewiselinear_forced.json"),
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40.json"),
                'enable_transform_score': False,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.053996580527295335, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 14.98274738501623, places=4)

    def test_run_bootstrap_vmaf_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.json'),
            },
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.json'),
                'enable_transform_score': True,
            },
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path('vmaf_rb_v0.6.2', 'vmaf_rb_v0.6.2.json'),
            },
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path('vmaf_rb_v0.6.3', 'vmaf_rb_v0.6.3.json'),
            },
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 75.41106825492248, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 99.95804893252175, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 74.93633669028638, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 99.93908291255723, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 1.31580415378762, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 0.09930398700617331, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_ci95_low_score'], 72.95957599695008, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_ci95_low_score'], 91.11567337246434, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_ci95_high_score'], 77.36730937306012, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_ci95_high_score'], 100.0, places=4)

    def test_run_bootstrap_vmaf_runner_default_model_with_transform_score(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'enable_transform_score': True,
            }
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 91.70144407600776, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_bagging_score'], 91.38626993803041, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_bagging_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_stddev_score'], 0.8796499060403885, places=4)
        self.assertAlmostEqual(results[1]['BOOTSTRAP_VMAF_stddev_score'], 0.0, places=4)

    def test_run_bootstrap_vmaf_runner_10models(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test2.json'),
            },
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.json'),
            },
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=False)

        results = self.runner.results
        self.assertAlmostEqual(results[0]['NIQE_score'], 4.8656072348129422, places=4)
        self.assertAlmostEqual(results[1]['NIQE_score'], 2.9309929860778756, places=2)

    def test_run_vmaf_runner_with_4k_1d5H_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_4k_v0.6.1.json"),
            },
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 84.9506475312982, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_bootstrap_vmaf_runner_with_4k_1d5H_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path('vmaf_4k_rb_v0.6.2', 'vmaf_4k_rb_v0.6.2.json'),
            },
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2_test.json'),
            },
        )
        self.runner.run(parallelize=False)

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
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.6895391472210215, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.9576202275437821, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.97696930627404, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9829181243754922, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 1.3596959375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9807496975664337, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 1.3596959375, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_ansnr_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 92.52344867729687, places=3)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.30930978456455, places=4)

    def test_run_vmaf_runner_json_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1.json"),
            }
        )
        self.runner.run(parallelize=False)

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

    def test_run_bootstrap_vmaf_runner_default_model_json_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_float_b_v0.6.3.json"),
            }
        )
        self.runner.run(parallelize=False)

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

    def test_run_vmaf_runner_motion_force_zero(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('other_models', 'vmaf_v0.6.1mfz.json')}
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_ansnr_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 72.3205499536087, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 97.42843608965536, places=4)

    def test_run_vmaf_runner_neg_mode(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafnegQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFNEG_score'], 88.030463, places=4)  # 132.7329528948058

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.9837379749630343, places=4)
        with self.assertRaises(KeyError):
            _ = results[0]['VMAF_integer_feature_vif_scale0_egl_1_score']  # egl_1 not in feature name

    def test_run_vmaf_runner_float_neg_mode(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1neg.json")}
        )
        self.runner.run(parallelize=False)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 88.03295534339294, places=2)  # 132.7329528948058

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.9837379749630343, places=4)
        with self.assertRaises(KeyError):
            _ = results[0]['VMAF_feature_vif_scale0_egl_1_score']  # egl_1 not in feature name

    def test_run_vmaf_runner_nvd6(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_v0.6.1_nvd6.json')}
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9535583604166833, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 80.60147296308644, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_float_nvd6(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_nvd6.json')}
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9535689329913465, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 80.61670115719328, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_rdh540(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_v0.6.1_rdh540.json')}
        )

        # current implementation is limited by the 16-bit data pipeline, thus
        # cannot handle an angular frequency smaller than 1080p * 3H
        with self.assertRaises(AssertionError):
            self.runner.run(parallelize=False)

            results = self.runner.results

            self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.3636620710647402, places=4)
            self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7674952820232231, places=4)
            self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8631077727416296, places=4)
            self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9157200890843669, places=4)
            self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
            self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9179700081140197, places=4)

            self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.00000001415, places=4)
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 0.99999972612, places=4)
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 0.999999465724, places=4)
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 0.999999399683, places=4)
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

            self.assertAlmostEqual(results[0]['VMAF_score'], 73.28968543912883, places=4)
            self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_float_rdh540(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_rdh540.json')}
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9179700081140197, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 73.28968543912883, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_rdh2160_nvd1d5(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_v0.6.1_rdh2160_nvd1d5.json')}
        )
        self.runner.run(parallelize=False)

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

    def test_run_vmaf_runner_float_vifks3o2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_vifks3o2.json')},
            optional_dict2={'disable_avx': False},
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3837840319479306, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.857804947713165, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.9233334535128187, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9531892263362162, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 81.50635007192874, places=4)  # pyvmaf: 81.46718219339385
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_float_vifks24o10(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_vifks24o10.json')},
            optional_dict2={'disable_avx': False},
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.4054443114816349, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.9245720373397553, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.9604211024859047, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9750504554016111, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 84.6878842953436, places=3)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_float_vifks360o97(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_vifks360o97.json')},
            optional_dict2={'disable_avx': False},
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.4250520955733605, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.9583187986343104, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.976499720370497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9824524789568665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 85.97569894767425, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_float_vifks1o2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_vifks1o2.json')},
            optional_dict2={'disable_avx': False},
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.33224252285074446, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.5843307831368371, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.7290873047465741, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9384341698106438, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 75.26224138570674, places=2)  # pyvmaf: 75.14868810608671
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_float_vifks2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_vifks2.json')},
            optional_dict2={'disable_avx': False},
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.39720947799409806, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.902590363978247, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.9488381004744343, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9685473062944817, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9344888814000374, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 83.69301110293519, places=2)  # pyvmaf: 83.699460342183
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_float_vifks2o3(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_vifks2o3.json')},
            optional_dict2={'disable_avx': False},
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.34550238312796616, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.6617754812987201, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.7862132379297329, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.8933108841437846, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9344888814000374, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 73.12401678992882, places=2)  # pyvmaf: 72.97880576181906
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_flat_save_workfiles_fifo_true(self):

        ref_path, dis_path, asset, asset_original = set_default_flat_1920_1080_videos_for_testing()

        with self.assertRaises(AssertionError) as ctx:
            self.runner = VmafQualityRunner(
                [asset, asset_original],
                None, fifo_mode=True,
                delete_workdir=True,
                result_store=self.result_store,
                save_workfiles=True,
            )
            self.runner.run(parallelize=False)
        self.assertTrue('To save workfiles, FIFO mode cannot be true.' in str(ctx.exception))

    def test_run_vmaf_runner_with_param_neg(self):
        """
        By the rule in FeatureAssembler._get_fextractor_instance(), there is no feature_option_dict specified in the
        default model file vmaf_v0.6.1.json, so the fields in optional_dict kicks in.
        """

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'vif_enhn_gain_limit': 1.0, 'adm_enhn_gain_limit': 1.0},
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7648294563297106, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8585079136153025, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9092599837535159, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9298444431333371, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 75.07472948234982, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.94635786905758, places=4)

    def test_run_vmaf_runner_with_param_neg_and_model_mfz(self):
        """
        By the rule in FeatureAssembler._get_fextractor_instance(), use feature_option_dict specified in the
        default model file vmaf_v0.6.1mfz.json; the field 'vif_enhn_gain_limit' etc. has no effect.
        """

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'vif_enhn_gain_limit': 1.0, 'adm_enhn_gain_limit': 1.0,  # no effect
                'model_filepath': VmafConfig.model_path('other_models', 'vmaf_v0.6.1mfz.json')
            },
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 72.3205498755804, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 97.42843609144575, places=4)


class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmaf_quality_runner_version(self):
        self.assertEqual(VmafQualityRunner.VERSION, 'F0.2.7int-0.6.1')
        self.assertEqual(VmafQualityRunner.ALGO_VERSION, 4)


class QualityRunnerResultStoreTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        self.runner1.remove_results()
        self.runner2.remove_results()
        super().tearDown()

    def test_quality_runner_with_different_models(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner1 = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_rdh540.json')}
        )
        self.runner1.run(parallelize=False)
        results1 = self.runner1.results

        self.runner2 = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_nvd6.json')}
        )
        self.runner2.run(parallelize=False)
        results2 = self.runner2.results

        self.assertAlmostEqual(results1[0]['VMAF_score'], 73.28968543912883, places=4)
        self.assertAlmostEqual(results1[1]['VMAF_score'], 99.946416604585025, places=4)

        self.assertAlmostEqual(results2[0]['VMAF_score'], 80.61670115719328, places=4)
        self.assertAlmostEqual(results2[1]['VMAF_score'], 99.946416604585025, places=4)


class QualityRunnerSaveWorkfilesTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
        super().tearDown()

    def test_run_vmaf_runner_flat_save_workfiles(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=self.result_store,
            save_workfiles=True,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.66890511746402, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.94642662500576, places=4)

    def test_run_psnr_runner_flat_save_workfiles(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = PsnrQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=self.result_store,
            save_workfiles=True,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166668, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
