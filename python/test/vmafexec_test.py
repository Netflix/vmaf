from __future__ import absolute_import

import unittest

import vmaf
from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafexecQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from test.testutil import set_default_576_324_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing, set_default_576_324_10bit_videos_for_testing_b, \
    set_default_576_324_12bit_videos_for_testing, set_default_576_324_16bit_videos_for_testing
from vmaf.tools.misc import MyTestCase

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class VmafexecQualityRunnerTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass
        super().tearDown()

    def test_run_vmafexec_runner_matched_to_vmafossexec(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'],0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.946416604585025, places=4)

    def test_run_vmafexec_runner_float_fex(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
                'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1.json")
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.3634208125, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.7666474166666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.8628533333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9159719583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'],0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'],0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 76.68425579166666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.94641666666666, places=4)

    def test_run_vmafexec_runner_motion_force_zero(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'motion_force_zero': True,
                'model_filepath': vmaf.model_path("vmaf_v0.6.1.json")
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 72.32054995833333, places=4)  # 76.68425579166666
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 97.42843597916665, places=4)  # 99.94641666666666

        self.assertEqual(len(results[0]['VMAFEXEC_motion2_force_0_scores']), 48)
        self.assertEqual(len(results[1]['VMAFEXEC_motion2_force_0_scores']), 48)

    def test_run_vmafexec_runner_motion_force_zero2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("other_models", "vmaf_v0.6.1mfz.json")
            }
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 72.32054995833333, places=4)  # 76.68425579166666
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 97.42843597916665, places=4)  # 99.94641666666666

        self.assertEqual(len(results[0]['VMAFEXEC_motion2_force_0_scores']), 48)
        self.assertEqual(len(results[1]['VMAFEXEC_motion2_force_0_scores']), 48)

    def test_run_vmafexec_runner_fixed_psnr(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'psnr': True,
                'no_prediction': True,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_psnr_y_score'], 30.755063979166668, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_psnr_cb_score'], 38.4494410625, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_psnr_cr_score'], 40.99191027083334, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_psnr_y_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_psnr_cb_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_psnr_cr_score'], 60.0, places=4)

    def test_run_vmafexec_runner_n_threads(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
                'psnr': True,
                'n_threads': 4,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ms_ssim_score'], 0.9632406874999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_psnr_y_score'], 30.755063979166668, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_psnr_cb_score'], 38.4494410625, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_psnr_cr_score'], 40.99191027083334, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ms_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_psnr_y_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_psnr_cb_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_psnr_cr_score'], 60.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 76.66890489583334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.94641666666666, places=4)

    def test_run_vmafexec_runner_set_custom_models(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'models': [
                    'path={}:name=custom_vmaf_0'.format(VmafConfig.model_path("vmaf_float_v0.6.1.json")),
                    'path={}:name=custom_vmaf_1'.format(VmafConfig.model_path("vmaf_float_v0.6.1.json")),
                ]
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_custom_vmaf_0_score'], 76.68425579166666, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_custom_vmaf_1_score'], 76.68425579166666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_custom_vmaf_0_score'], 99.94641666666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_custom_vmaf_1_score'], 99.94641666666666, places=4)

    def test_run_vmafexec_runner_set_custom_models_enable_transform(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'models': [
                    'path={}:name=standvmaf'.format(VmafConfig.model_path("vmaf_float_v0.6.1.json")),
                    'path={}:name=phonevmaf:enable_transform'.format(VmafConfig.model_path("vmaf_float_v0.6.1.json")),
                ]
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_standvmaf_score'], 76.68425579166666, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_phonevmaf_score'], 92.53270047916665, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_standvmaf_score'], 99.94641666666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_phonevmaf_score'], 100.0, places=4)

    def test_run_vmafexec_runner_disable_avx(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'disable_avx': True
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.946416604585025, places=4)

    def test_run_parallel_vmafexec_runner_with_repeated_assets(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 76.66890519623612, places=3)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.946416666666664, places=4)
        self.assertAlmostEqual(results[2]['VMAFEXEC_score'], 76.66890519623612, places=3)
        self.assertAlmostEqual(results[3]['VMAFEXEC_score'], 76.66890519623612, places=3)

    def test_run_vmafexec_runner_yuv422p10le(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'],0.3636620625, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.7674953125, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.8631078125, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9157200833333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_psnr_score'], 30.780577083333331, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 76.66890489583334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.94641666666666, places=4)

    def test_run_vmafexec_runner_yuv420p10le_b(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing_b()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.4330893333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.830613, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.9072123333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.945896, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_psnr_score'], 32.57143333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ssim_score'], 0.8978630000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ms_ssim_score'], 0.9747490000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 82.56523033333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.142826, places=4)

    def test_run_vmafexec_runner_yuv420p12le(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'],0.4330893333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.830613, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.9072123333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.945896, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_psnr_score'], 32.577818, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ssim_score'], 0.8978630000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ms_ssim_score'], 0.9747490000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_psnr_score'], 84.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 82.56523033333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.142826, places=4)

    def test_run_vmafexec_runner_yuv420p16le(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'],0.4330893333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.830613, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.9072123333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.945896, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_psnr_score'], 32.579806000000005, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ssim_score'], 0.8978630000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ms_ssim_score'], 0.9747490000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_psnr_score'], 108.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 82.56523033333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.142826, places=4)

    def test_run_vmafexec_runner_yuv420p10le_sparks(self):

        ref_path = VmafConfig.test_resource_path("yuv", "sparks_ref_480x270.yuv42010le.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "sparks_dis_480x270.yuv42010le.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 480, 'height': 270,
                                  'yuv_type': 'yuv420p10le'})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 480, 'height': 270,
                                  'yuv_type': 'yuv420p10le'})

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.9240746, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.9968371999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.9987575999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9993221999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 0.7523685999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9981770000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_psnr_score'], 48.81622, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ssim_score'], 0.99566, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_ms_ssim_score'], 0.9993778000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'], 0.9999990000000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'], 0.9999990000000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.9999990000000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 0.7523685999999999, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 97.90069380000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 98.47175940000001, places=4)

    def test_run_vmafexec_runner_float_moment(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_moment': True,
                'no_prediction': True,
            }
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_float_moment_ref1st_score'], 59.788567354166666, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_moment_dis1st_score'], 61.332006625, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_moment_ref2nd_score'], 4696.668388125, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_float_moment_dis2nd_score'], 4798.659574041666, places=4)

    def test_run_vmafexec_runner_akiyo_multiply(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 1.116691, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 1.052403, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 1.070149, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 1.072518, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 1.072512, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 132.732952, places=3)  # 132.78849246495625

    def test_run_vmafexec_runner_akiyo_multiply_with_feature_enhn_gain_limit(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'adm_enhn_gain_limit': 1.0, 'vif_enhn_gain_limit': 1.0}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_egl_1_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_egl_1_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_egl_1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_egl_1_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_egl_1_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 88.030463, places=4)  # 132.78849246495625

    def test_run_vmafexec_runner_akiyo_multiply_with_feature_enhn_gain_limit_custom(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'adm_enhn_gain_limit': 1.2, 'vif_enhn_gain_limit': 1.1}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_egl_1.2_score'], 1.116595, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_egl_1.1_score'], 1.029842, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_egl_1.1_score'], 1.046578, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_egl_1.1_score'], 1.04852, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_egl_1.1_score'], 1.04892, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 129.474226, places=3)  # 132.78849246495625

    def test_run_vmafexec_runner_akiyo_multiply_disable_enhn_gain(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'disable_enhn_gain': True}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_egl_1_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_egl_1_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_egl_1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_egl_1_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_egl_1_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 88.030463, places=4)  # 132.78849246495625

    def test_run_vmafexec_runner_akiyo_multiply_no_enhn_gain_model(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_v0.6.1neg.json")}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_egl_1_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_egl_1_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_egl_1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_egl_1_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_egl_1_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 88.030463, places=4)  # 132.78849246495625

    def test_run_vmafexec_runner_akiyo_multiply_no_enhn_gain_model_and_cmd_options(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_v0.6.1neg.json"),
                'adm_enhn_gain_limit': 1.2 }
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_egl_1.2_score'], 1.116595, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_egl_1_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_egl_1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_egl_1_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_egl_1_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 122.804272, places=3)  # 132.78849246495625

    def test_run_vmafexec_runner_akiyo_multiply_no_enhn_gain_model_and_cmd_options_illegal(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1neg.json"),
                           'adm_enhn_gain_limit': 0.9}
        )
        with self.assertRaises(AssertionError,
                               msg="adm_enhn_gain_limit is below 1 but the exception is not raised"):
            self.runner.run(parallelize=False)

    def test_run_vmafexec_runner_akiyo_multiply_no_enhn_gain_model_and_cmd_options_illegal2(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1neg.json"),
                           'vif_enhn_gain_limit': 0.9}
        )
        with self.assertRaises(AssertionError,
                               msg="vif_enhn_gain_limit is below 1 but the exception is not raised"):
            self.runner.run(parallelize=False)

    def test_run_vmafexec_runner_akiyo_multiply_with_feature_enhn_gain_limit_b_v063(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True,
                           'model_filepath': VmafConfig.model_path("vmaf_float_b_v0.6.3", "vmaf_float_b_v0.6.3.json"),
                           'adm_enhn_gain_limit': 1.0, 'vif_enhn_gain_limit': 1.0}
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_egl_1_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_egl_1_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_egl_1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_egl_1_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_egl_1_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 88.4895, places=2)  # 88.032956

    def test_run_vmafexec_runner_use_default_built_in_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'use_default_built_in_model': True
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale2_score'],0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAFEXEC_score'], 99.946416604585025, places=4)

    def test_run_vmafexec_runner_akiyo_multiply_4k_model(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_4k_v0.6.1.json")}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.3636620625, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.7674953125, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.8631078125, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.9157200833333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm2_score'], 0.9345148541666667, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 84.95064735416668, places=4)

    def test_run_vmaf_runner_with_transform_score(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'models': [
                    'path={}:name=vmat:enable_transform'.format(VmafConfig.test_resource_path("test_model_transform_add40.json")),
                ],
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_vmat_score'], 32.757433750978919, places=4)

    def test_run_vmaf_runner_with_transform_score_2(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'models': [
                    'path={}:name=vmat:enable_transform:disable_clip'.format(VmafConfig.test_resource_path("test_model_transform_add40_outltein.json")),
                ],
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_vmat_score'], -7.2425662490210838, places=4)

    def test_run_vmaf_runner_with_transform_score3(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'models': [
                    'path={}:name=vmat:enable_transform'.format(VmafConfig.test_resource_path("test_model_transform_add40_piecewiselinear.json")),
                ],
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_vmat_score'], 8.262602639723815, places=4)

    def test_run_vmaf_runner_with_transform_score4(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'models': [
                    'path={}:name=vmat'.format(VmafConfig.test_resource_path("test_model_transform_add40_piecewiselinear_forced.json")),
                ],
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFEXEC_vmat_score'], 8.262602639723815, places=4)

    def test_run_vmaf_runner_with_transform_score_both_specified(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafexecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'models': [
                    'path={}:name=vmat:enable_transform'.format(VmafConfig.test_resource_path("test_model_transform_add40_piecewiselinear_forced.json")),
                ],
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_vmat_score'], 8.262602639723815, places=4)


class VmafexecQualityRunnerSubsamplingTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runner0'):
            self.runner0.remove_results()
        if hasattr(self, 'runner'):
            self.runner.remove_results()
        super().tearDown()

    def test_run_vmafexec_runner_with_subsample2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        subsample = 5

        self.runner0 = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={}
        )
        self.runner0.run()
        results0 = self.runner0.results

        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'subsample': subsample}
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        for i in range(48):
            if i % subsample == 0:
                self.assertAlmostEqual(results0[0]['VMAFEXEC_scores'][i], results[0]['VMAFEXEC_scores'][i // subsample], places=7)
                self.assertAlmostEqual(results0[1]['VMAFEXEC_scores'][i], results[1]['VMAFEXEC_scores'][i // subsample], places=7)


class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmafexec_quality_runner_version(self):
        self.assertEqual(VmafexecQualityRunner.VERSION, 'F0.2.7-0.6.1')
        self.assertEqual(VmafexecQualityRunner.ALGO_VERSION, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
