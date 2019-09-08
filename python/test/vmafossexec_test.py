from __future__ import absolute_import

import unittest

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore

from .testutil import set_default_576_324_videos_for_testing

__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class VmafossexecQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_vmafossexec_runner(self):
        print('test on running VMAFOSSEXEC runner...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.9079192708333332, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.8939565625, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.9301004166666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.9650352708333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion_score'], 4.0498256249999995, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion_score'], 4.0498256249999995, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_thread(self):
        print('test on running VMAFOSSEXEC runner with thread...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'thread': 3}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_061_additional_model(self):
        print('test on running VMAFOSSEXEC runner with 0.6.1 as additional model...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'additional_models': {"some_model": {"model_path": VmafConfig.model_path("vmaf_v0.6.1.pkl")}}}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_some_model_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_some_model_score'], 99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_061_additional_model_enable_transform(self):
        print('test on running VMAFOSSEXEC runner with 0.6.1 as additional model + enable_transform...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'additional_models': {"some_model": {"model_path": VmafConfig.model_path("vmaf_v0.6.1.pkl"),
                                                                "enable_transform": "1"}}}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_some_model_score'], 92.54239166666667, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_some_model_score'], 100.0, places=4)

    def test_run_vmafossexec_runner_with_061_additional_models(self):
        print('test on running VMAFOSSEXEC runner with 0.6.1 as additional models...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'additional_models': {"some_model": {"model_path": VmafConfig.model_path("vmaf_v0.6.1.pkl")},
                                                 "other_model": {"model_path": VmafConfig.model_path("vmaf_v0.6.1.pkl")}}}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_some_model_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_other_model_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_some_model_score'], 99.946416604585025, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_other_model_score'], 99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_061_additional_models_one_is_transformed(self):
        print('test on running VMAFOSSEXEC runner with 0.6.1 as additional models...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'additional_models': {"some_model": {"model_path": VmafConfig.model_path("vmaf_v0.6.1.pkl")},
                                                 "other_model": {"model_path": VmafConfig.model_path("vmaf_v0.6.1.pkl"),
                                                                 "enable_transform": "1"}}}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_some_model_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_other_model_score'], 92.54239166666667, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_some_model_score'], 99.946416604585025, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_other_model_score'], 100.0, places=4)

    def test_run_vmafossexec_runner_with_4k_additional_model(self):
        print('test on running VMAFOSSEXEC runner with 4k as additional model...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'additional_models': {"four_k": {"model_path": VmafConfig.model_path("vmaf_4k_v0.6.1.pkl")}}}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_four_k_score'], 84.98462083333332, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_four_k_score'], 100.0, places=4)

    def test_run_vmafossexec_runner_default_and_additional_model_use_ci(self):
        print('test on running VMAFOSSEXEC runner with additional models and ci...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_b_v0.6.3', 'vmaf_b_v0.6.3.pkl'),
                           'additional_models': {'vmaf_2': {"model_path": VmafConfig.model_path('vmaf_b_v0.6.3', 'vmaf_b_v0.6.3.pkl'),
                                                            "enable_conf_interval": "1"}},
                           'ci': True}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 75.44304862545658, places=3)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bagging_score'], 74.96365833333334, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_stddev_score'], 1.3128927083333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_ci95_low_score'], 72.98503333333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_ci95_high_score'], 77.38652708333333, places=3)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_bagging_score'], 74.96365833333334, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_stddev_score'], 1.3128927083333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_ci95_low_score'], 72.98503333333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_ci95_high_score'], 77.38652708333333, places=3)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.95804791666667, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_bagging_score'], 99.93908333333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_stddev_score'], 0.09930395833333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_ci95_low_score'], 91.11520624999999, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_ci95_high_score'], 100.0, places=3)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_2_bagging_score'], 99.93908333333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_2_stddev_score'], 0.09930395833333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_2_ci95_low_score'], 91.11520624999999, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_2_ci95_high_score'], 100.0, places=3)

    def test_run_vmafossexec_runner_only_additional_models_use_ci(self):
        print('test on running VMAFOSSEXEC runner with additional models and ci...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v0.6.1.pkl'),
                           'additional_models': {'vmaf_2': {"model_path": VmafConfig.model_path('vmaf_b_v0.6.3', 'vmaf_b_v0.6.3.pkl'), "enable_conf_interval": "1"},
                                                 'vmaf_rb_v0.6.2': {"model_path": VmafConfig.model_path('vmaf_rb_v0.6.2', 'vmaf_rb_v0.6.2.pkl'), "enable_conf_interval": "1"},
                                                 'vmaf_rb_v0.6.3': {"model_path": VmafConfig.model_path('vmaf_rb_v0.6.3', 'vmaf_rb_v0.6.3.pkl'), "enable_conf_interval": "1"},
                                                 'vmaf_4k_rb_v0.6.2': {"model_path": VmafConfig.model_path('vmaf_4k_rb_v0.6.2', 'vmaf_4k_rb_v0.6.2.pkl'), "enable_conf_interval": "1"}}}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.9079192708333332, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.8939565625, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.9301004166666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.9650352708333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion_score'], 4.0498256249999995, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.69926875, places=3)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_bagging_score'], 74.96365833333334, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_stddev_score'], 1.3128927083333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_ci95_low_score'], 72.98503333333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_ci95_high_score'], 77.38652708333333, places=3)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_4k_rb_v0.6.2_bagging_score'], 83.98690625, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_4k_rb_v0.6.2_stddev_score'], 0.9290704791666666, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_4k_rb_v0.6.2_ci95_low_score'], 82.76451250000001, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_4k_rb_v0.6.2_ci95_high_score'], 85.85374791666668, places=3)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.94641666666666, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_2_bagging_score'], 99.93908333333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_2_stddev_score'], 0.09930395833333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_2_ci95_low_score'], 91.11520624999999, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_2_ci95_high_score'], 100.0, places=3)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_4k_rb_v0.6.2_bagging_score'], 99.97613958333334, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_4k_rb_v0.6.2_stddev_score'], 0.036273541666666666, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_4k_rb_v0.6.2_ci95_low_score'], 97.65394583333334, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_4k_rb_v0.6.2_ci95_high_score'], 100.0, places=3)

        self.assertEquals(results[0]['VMAFOSSEXEC_vmaf_rb_v0.6.3_bootstrap_0001_scores'], results[0]['VMAFOSSEXEC_vmaf_rb_v0.6.2_bootstrap_0001_scores'])
        self.assertEquals(results[0]['VMAFOSSEXEC_vmaf_rb_v0.6.3_bootstrap_0019_scores'], results[0]['VMAFOSSEXEC_vmaf_rb_v0.6.2_bootstrap_0019_scores'])
        self.assertEquals(results[1]['VMAFOSSEXEC_vmaf_rb_v0.6.3_bootstrap_0019_scores'], results[1]['VMAFOSSEXEC_vmaf_rb_v0.6.2_bootstrap_0019_scores'])

    def test_run_vmafossexec_runner_additional_models_use_ci_et_mix_match(self):
        print('test on running VMAFOSSEXEC runner with additional models and ci...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v0.6.1.pkl'),
                           'additional_models': {'vmaf_061_wo_ci_wo_et': {'model_path': VmafConfig.model_path('vmaf_v0.6.1.pkl'), 'enable_conf_interval': '0', 'enable_transform': '0'},
                                                 'vmaf_061_wo_ci_w_et': {'model_path': VmafConfig.model_path('vmaf_v0.6.1.pkl'), 'enable_conf_interval': '0', 'enable_transform': '1'},
                                                 'vmaf_b063_w_ci_wo_et': {'model_path': VmafConfig.model_path('vmaf_b_v0.6.3', 'vmaf_b_v0.6.3.pkl'), "enable_conf_interval": '1', 'enable_transform': '0'},
                                                 'vmaf_b063_w_ci_w_et': {'model_path': VmafConfig.model_path('vmaf_b_v0.6.3', 'vmaf_b_v0.6.3.pkl'),"enable_conf_interval": '1', 'enable_transform': '1'}}
                           }
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.69926875, places=3)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_061_wo_ci_wo_et_score'], 76.69926875, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_061_wo_ci_w_et_score'], 92.54239166666667, places=3)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_b063_w_ci_wo_et_bagging_score'], 74.96365833333334, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_b063_w_ci_wo_et_stddev_score'], 1.3128927083333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_b063_w_ci_wo_et_ci95_low_score'], 72.98503333333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_b063_w_ci_wo_et_ci95_high_score'], 77.38652708333333, places=3)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_b063_w_ci_w_et_bagging_score'], 91.40493125, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_b063_w_ci_w_et_stddev_score'], 0.8773241874999999, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_b063_w_ci_w_et_ci95_low_score'], 90.05703125000001, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_b063_w_ci_w_et_ci95_high_score'], 92.98213541666667, places=3)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.94641666666666, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_061_wo_ci_wo_et_score'], 99.94641666666666, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_061_wo_ci_w_et_score'], 100.0, places=3)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_b063_w_ci_wo_et_bagging_score'], 99.93908333333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_b063_w_ci_wo_et_stddev_score'], 0.09930395833333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_b063_w_ci_wo_et_ci95_low_score'], 91.11520624999999, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_b063_w_ci_wo_et_ci95_high_score'], 100.0, places=3)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_b063_w_ci_w_et_bagging_score'], 100.0, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_b063_w_ci_w_et_stddev_score'], 0.0, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_b063_w_ci_w_et_ci95_low_score'], 99.94739583333332, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_b063_w_ci_w_et_ci95_high_score'], 100.0, places=3)

    def test_run_vmafossexec_runner_with_additional_model_use_ci_et(self):
        print('test on running VMAFOSSEXEC runner with additional models and ci and enable transform...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_b_v0.6.3', 'vmaf_b_v0.6.3.pkl'),
                           'additional_models': {'vmaf_2': {"model_path": VmafConfig.model_path('vmaf_b_v0.6.3', 'vmaf_b_v0.6.3.pkl'),
                                                            "enable_conf_interval": "1",
                                                            "enable_transform": "1"}},
                           'ci': True,
                           'enable_transform_score': True}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bagging_score'], 91.40493125, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_bagging_score'], 91.40493125, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_ci95_low_score'], 90.05703125000001, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_2_ci95_low_score'], 90.05703125000001, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_bagging_score'], 100.0, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_2_bagging_score'], 100.0, places=3)

    def test_additional_model_json_string(self):
        print('test additional model json string creation...')
        self.assertEquals(VmafossExecQualityRunner
                          .get_json_additional_model_string({}), "")
        self.assertEquals(VmafossExecQualityRunner.get_json_additional_model_string({"model A": {"model_path": "/someA/pthA/forA/A.pkl"}}),
                          '{\\"model A\\"\\:{\\"model_path\\"\\:\\"/someA/pthA/forA/A.pkl\\"}}')
        self.assertEquals(VmafossExecQualityRunner.get_json_additional_model_string({"model A": {"model_path": "/someA/pthA/forA/A.pkl"}, "model B": {"model_path": "/someB/pthB/forB/B.pkl"}}),
                          '{\\"model A\\"\\:{\\"model_path\\"\\:\\"/someA/pthA/forA/A.pkl\\"}\\,\\"model B\\"\\:{\\"model_path\\"\\:\\"/someB/pthB/forB/B.pkl\\"}}')
        self.assertEquals(VmafossExecQualityRunner.get_json_additional_model_string({"model B": {"model_path": "/someB/pthB/forB/B.pkl"}, "model A": {"model_path": "/someA/pthA/forA/A.pkl"}}),
                          '{\\"model A\\"\\:{\\"model_path\\"\\:\\"/someA/pthA/forA/A.pkl\\"}\\,\\"model B\\"\\:{\\"model_path\\"\\:\\"/someB/pthB/forB/B.pkl\\"}}')
        self.assertEquals(VmafossExecQualityRunner.get_json_additional_model_string({"model A": {"model_path": "/someA/pthA/forA/A.pkl", "enable_transform": "1"}, "model B": {"model_path": "/someB/pthB/forB/B.pkl"}}),
                          '{\\"model A\\"\\:{\\"enable_transform\\"\\:\\"1\\"\\,\\"model_path\\"\\:\\"/someA/pthA/forA/A.pkl\\"}\\,\\"model B\\"\\:{\\"model_path\\"\\:\\"/someB/pthB/forB/B.pkl\\"}}')
        self.assertEquals(VmafossExecQualityRunner.get_json_additional_model_string({"model A": {"model_path": "/someA/pthA/forA/A.pkl", "enable_transform": "1"}, "model B": {"model_path": "/someB/pthB/forB/B.pkl", "disable_clip": "0"}}),
                          '{\\"model A\\"\\:{\\"enable_transform\\"\\:\\"1\\"\\,\\"model_path\\"\\:\\"/someA/pthA/forA/A.pkl\\"}\\,\\"model B\\"\\:{\\"disable_clip\\"\\:\\"0\\"\\,\\"model_path\\"\\:\\"/someB/pthB/forB/B.pkl\\"}}')

    def test_run_vmafossexec_runner_with_subsample(self):
        print('test on running VMAFOSSEXEC runner with subsample...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'subsample': 5}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.954390000000018, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.742800000000003, places=4)

    def test_run_vmafossexec_runner_psnr_color(self):
        print('test on running VMAFOSSEXEC runner with color ...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'use_color': True}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.69926875, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.94641666666666, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.755066666666664, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_u_score'], 38.449447916666664, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_v_score'], 40.9919125, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_u_score'], 60.0, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_v_score'], 60.0, places=3)

    def test_run_vmafossexec_runner_psnr_color_single_thread(self):
        print('test on running VMAFOSSEXEC runner with color using one thread ...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'use_color': True, 'thread': 1}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.69926875, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.94641666666666, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.755066666666664, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_u_score'], 38.449447916666664, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_v_score'], 40.9919125, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_u_score'], 60.0, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_v_score'], 60.0, places=3)

    def test_run_vmafossexec_runner_psnr_color_two_threads(self):
        print('test on running VMAFOSSEXEC runner with color using two threads ...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'use_color': True, 'thread': 2}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.69926875, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.94641666666666, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.755066666666664, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_u_score'], 38.449447916666664, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_v_score'], 40.9919125, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_u_score'], 60.0, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_v_score'], 60.0, places=3)

    def test_run_vmafossexec_runner_with_phone_score(self):
        print('test on running VMAFOSSEXEC runner with phone score...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 92.542390144364546, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 100.0, places=4)

    def test_run_vmafossexec_runner_norm_type_none(self):
        print('test on running VMAFOSSEXEC runner with norm type none...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath':VmafConfig.model_path("other_models", "nflxtrain_norm_type_none.pkl"),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.9079192708333332, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.8939565625, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.9301004166666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.9650352708333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 74.253349625150562, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 77.996338095161946, places=4)

    def test_run_vmafossexec_runner_yuv422p10le(self):
        print('test on running VMAFOSSEXEC runner on 10 bit le...')
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv422p10le.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv422p10le.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'yuv_type':'yuv422p10le'})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'yuv_type':'yuv422p10le'})

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.780577083333331, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_transform_score(self):
        print('test on running VMAFOSSEXEC runner with score transforming...')
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 32.757433333333331, places=4)

    def test_run_vmafossexec_runner_with_transform_score_2(self):
        print('test on running VMAFOSSEXEC runner with score transforming...')
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], -7.2425766666666673, places=4)

    def test_run_vmafossexec_runner_with_transform_score_disabled(self):
        print('test on running VMAFOSSEXEC runner with score transforming disabled...')
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'model_filepath':VmafConfig.test_resource_path("test_model_transform_add40.pkl"),
                'enable_transform_score':False,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 0.0, places=4)

    def test_run_vmafossexec_runner_with_transform_for_phone(self):
        print('test on running VMAFOSSEXEC runner with transform for phone...')
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'enable_transform_score':True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.053996566666666669, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 14.982840000000001, places=4)

    def test_run_vmafossexec_runner_with_phone_model(self):
        print('test on running VMAFOSSEXEC runner with transform for phone...')
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'phone_model': True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.053996566666666669, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 14.982840000000001, places=4)

    def test_run_vmafossexec_runner_disable_avx_precise(self):
        print('test on running VMAFOSSEXEC runner disabling AVX (precise)...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_avx': True}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953527083333337, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.96324068749999991, places=5)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.999999958333, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.999999416667, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953527083333337, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=5)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699306249999992, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416666666664, places=5)

    def test_run_vmafossexec_runner_enable_avx_precise(self):
        print('test on running VMAFOSSEXEC runner enabling AVX (precise)...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.36342081250000002, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.76664741666666669, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.86285333333333336, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.91597195833333345, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953527083333337, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.96324068749999991, places=5)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.99999985416666659, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.99999960416666667, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.99999914583333338, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953527083333337, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=5)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699266666666674, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416666666664, places=5)

    def test_run_vmafossexec_runner_with_motion2(self):
        print('test on running VMAFOSSEXEC runner with motion2 feature...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath':VmafConfig.test_resource_path("test_motion2.pkl")
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953522916666672, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953522916666672, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 78.532525000000007, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 97.089554166666673, places=4)

    def test_run_vmafossexec_runner_with_ci(self):
        print('test on running VMAFOSSEXEC runner with conf interval...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_rb_v0.6.3", "vmaf_rb_v0.6.3.pkl"),
                'ci': True
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 75.443043750000001, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.958047916666672, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bagging_score'], 73.10273541666668, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_bagging_score'], 99.79000416666668, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_stddev_score'], 1.1991330833333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_stddev_score'], 1.3028828125, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_ci95_low_score'], 70.82471875, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_ci95_low_score'], 94.79667083333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_ci95_high_score'], 74.85038125, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_ci95_high_score'], 99.99736666666666, places=4)

        # per model score checks
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bootstrap_0001_score'], 73.26853333333334, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bootstrap_0002_score'], 70.38517916666667, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bootstrap_0003_score'], 71.59264583333334, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bootstrap_0020_score'], 73.15570625, places=3)

    def test_run_vmafossexec_runner_with_ci_and_custom_model(self):
        print('test on running VMAFOSSEXEC runner with conf interval and custom model...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.pkl'),
                'ci': True
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 75.443043750000001, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.958047916666672, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bagging_score'], 75.13012623785923, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_bagging_score'], 99.96504855577571, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_stddev_score'], 0.6812993325967104, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_stddev_score'], 0.03947607207290399, places=4)

    def test_run_vmafossexec_runner_with_ci_and_phone_model(self):
        print('test on running VMAFOSSEXEC runner with conf interval and phone model...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_rb_v0.6.3", "vmaf_rb_v0.6.3.pkl"),
                'phone_model': True,
                'ci': True,
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 91.723012127641823, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bagging_score'], 90.13159583333334, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_bagging_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_stddev_score'], 0.8371132083333332, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_stddev_score'], 0.0, places=4)

        # per model score checks
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bootstrap_0001_score'], 90.25032499999999, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bootstrap_0002_score'], 88.18534583333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bootstrap_0003_score'], 89.04952291666666, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bootstrap_0020_score'], 90.16633958333334, places=3)


class VmafossexecQualityRunnerSubsamplingTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner0'):
            self.runner0.remove_results()
        if hasattr(self, 'runner'):
            self.runner.remove_results()

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_vmafossexec_runner_with_subsample2(self):
        print('test on running VMAFOSSEXEC runner with subsample2...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        subsample = 5

        self.runner0 = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={}
        )
        self.runner0.run()
        results0 = self.runner0.results

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'subsample': subsample}
        )
        self.runner.run()
        results = self.runner.results

        for i in range(48):
            if i % subsample == 0:
                self.assertAlmostEqual(results0[0]['VMAFOSSEXEC_scores'][i], results[0]['VMAFOSSEXEC_scores'][i // subsample], places=7)
                self.assertAlmostEqual(results0[1]['VMAFOSSEXEC_scores'][i], results[1]['VMAFOSSEXEC_scores'][i // subsample], places=7)


class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmafossexec_quality_runner_version(self):
        self.assertEquals(VmafossExecQualityRunner.VERSION, 'F0.2.4c-0.6.1')
        self.assertEquals(VmafossExecQualityRunner.ALGO_VERSION, 2)


class ParallelQualityRunnerTest(unittest.TestCase):

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runners'):
            for runner in self.runners:
                runner.remove_results()
            pass

    def test_run_parallel_vmafossexec_runner_with_repeated_assets(self):
        print('test on running VMAFOSSEXEC quality runner in parallel with repeated assets...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699266666666674, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416666666664, places=4)
        self.assertAlmostEqual(results[2]['VMAFOSSEXEC_score'], 76.699266666666674, places=3)
        self.assertAlmostEqual(results[3]['VMAFOSSEXEC_score'], 76.699266666666674, places=3)


if __name__ == '__main__':
    unittest.main()
