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

    def test_run_bootstrap_vmaf_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.json'),
            },
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
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.json'),
                'enable_transform_score': True,
            },
        )
        self.runner.run(parallelize=True)

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
                'model_filepath': VmafConfig.model_path('vmaf_rb_v0.6.2', 'vmaf_rb_v0.6.2.json'),
            },
        )
        self.runner.run(parallelize=True)

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
                'model_filepath': VmafConfig.model_path('vmaf_rb_v0.6.3', 'vmaf_rb_v0.6.3.json'),
            },
        )
        self.runner.run(parallelize=True)

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
        self.runner.run(parallelize=True)

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
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'enable_transform_score': True,
            }
        )
        self.runner.run(parallelize=True)

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
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test2.json'),
            },
        )
        self.runner.run(parallelize=True)

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
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.json'),
            },
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
        self.runner.run(parallelize=True)

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
                'model_filepath': VmafConfig.model_path("vmaf_4k_v0.6.1.json"),
            },
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 84.9506475312982, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_bootstrap_vmaf_runner_with_4k_1d5H_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path('vmaf_4k_rb_v0.6.2', 'vmaf_4k_rb_v0.6.2.json'),
            },
        )
        self.runner.run(parallelize=True)

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
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2_test.json'),
            },
        )
        self.runner.run(parallelize=True)

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
        self.runner.run(parallelize=True)

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

    def test_run_bootstrap_vmaf_runner_default_model_json_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = BootstrapVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_float_b_v0.6.3.json"),
            }
        )
        self.runner.run(parallelize=True)

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
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('other_models', 'vmaf_v0.6.1mfz.json')}
        )
        self.runner.run(parallelize=True)

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
