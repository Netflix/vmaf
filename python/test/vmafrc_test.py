from __future__ import absolute_import

import unittest

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafrcQualityRunner, VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from test.testutil import set_default_576_324_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing, set_default_576_324_10bit_videos_for_testing_b, \
    set_default_576_324_12bit_videos_for_testing, set_default_576_324_16bit_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class VmafrcQualityRunnerTest(unittest.TestCase):

    def setUp(self):
        self.result_store = FileSystemResultStore()
        self.verificationErrors = []
        self.maxDiff = None

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass
        self.assertEqual([], self.verificationErrors)

    def test_run_vmafrc_runner_matched_to_vmafossexec(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'],0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.946416604585025, places=4)

    def test_run_vmafrc_runner_float_fex(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.3634208125, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7666474166666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8628533333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9159719583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'],0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'],0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.68425579166666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.94641666666666, places=4)

    def test_run_vmafrc_runner_motion_force_zero(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'motion_force_zero': True,
                'model_filepath': VmafConfig.model_path("vmaf_v0.6.1.json")
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        try: self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.3636620710647402, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7674952820232231, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8631077727416296, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9157200890843669, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 0.0, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345149030293786, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))

        try: self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'], 0.9999998541666666, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'], 0.9999996041666667, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999991458333334, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 0.0, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)
        except AssertionError as e: self.verificationErrors.append(str(e))

        try: self.assertAlmostEqual(results[0]['VMAFRC_score'], 72.32054995833333, places=4)  # 76.68425579166666
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[1]['VMAFRC_score'], 97.42843597916665, places=4)  # 99.94641666666666
        except AssertionError as e: self.verificationErrors.append(str(e))

        self.assertEqual(len(results[0]['VMAFRC_motion2_scores']), 48)
        self.assertEqual(len(results[1]['VMAFRC_motion2_scores']), 48)

    def test_run_vmafrc_runner_fixed_psnr(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_psnr_y_score'], 30.755063979166668, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_cb_score'], 38.4494410625, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_cr_score'], 40.99191027083334, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_psnr_y_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_psnr_cb_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_psnr_cr_score'], 60.0, places=4)

    def test_run_vmafrc_runner_n_threads(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9632406874999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_y_score'], 30.755063979166668, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_cb_score'], 38.4494410625, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_cr_score'], 40.99191027083334, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ms_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_psnr_y_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_psnr_cb_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_psnr_cr_score'], 60.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.66890489583334, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.94641666666666, places=4)

    def test_run_vmafrc_runner_set_custom_models(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_custom_vmaf_0_score'], 76.68425579166666, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_custom_vmaf_1_score'], 76.68425579166666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_custom_vmaf_0_score'], 99.94641666666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_custom_vmaf_1_score'], 99.94641666666666, places=4)

    def test_run_vmafrc_runner_set_custom_models_enable_transform(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'models': [
                    'path={}:name=custom_vmaf_0'.format(VmafConfig.model_path("vmaf_float_v0.6.1.json")),
                    'path={}:name=custom_vmaf_1:enable_transform'.format(VmafConfig.model_path("vmaf_float_v0.6.1.json")),
                ]
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_custom_vmaf_0_score'], 76.68425579166666, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_custom_vmaf_1_score'], 92.53270047916665, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_custom_vmaf_0_score'], 99.94641666666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_custom_vmaf_1_score'], 100.0, places=4)

    def test_run_vmafrc_runner_disable_avx(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.946416604585025, places=4)

    def test_run_parallel_vmafrc_runner_with_repeated_assets(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.66890519623612, places=3)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.946416666666664, places=4)
        self.assertAlmostEqual(results[2]['VMAFRC_score'], 76.66890519623612, places=3)
        self.assertAlmostEqual(results[3]['VMAFRC_score'], 76.66890519623612, places=3)

    def test_run_vmafrc_runner_yuv422p10le(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'],0.3636620625, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7674953125, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8631078125, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9157200833333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 30.780577083333331, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.66890489583334, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.94641666666666, places=4)

    def test_run_vmafrc_runner_yuv420p10le_b(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing_b()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.4330893333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.830613, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.9072123333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.945896, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 32.57143333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.8978630000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9747490000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 82.56523033333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.142826, places=4)

    def test_run_vmafrc_runner_yuv420p12le(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'],0.4330893333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.830613, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.9072123333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.945896, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 32.577818, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.8978630000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9747490000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_psnr_score'], 84.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 82.56523033333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.142826, places=4)

    def test_run_vmafrc_runner_yuv420p16le(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'],0.4330893333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.830613, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.9072123333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.945896, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 32.579806000000005, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.8978630000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9747490000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_psnr_score'], 108.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 82.56523033333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.142826, places=4)

    def test_run_vmafrc_runner_yuv420p10le_sparks(self):

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

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.9240746, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.9968371999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.9987575999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9993221999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 0.7523685999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9981770000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 48.81622, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.99566, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9993778000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'], 0.9999990000000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'], 0.9999990000000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999990000000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 0.7523685999999999, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 97.90069380000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 98.47175940000001, places=4)

    def test_run_vmafrc_compare_directly_with_ossexec_420_8bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
                'psnr': True,
            }
        )
        self.runner.run(parallelize=True)

        results_rc = self.runner.results

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale0_score'], results[0]['VMAFOSSEXEC_vif_scale0_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale1_score'], results[0]['VMAFOSSEXEC_vif_scale1_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale2_score'], results[0]['VMAFOSSEXEC_vif_scale2_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale3_score'], results[0]['VMAFOSSEXEC_vif_scale3_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_motion2_score'], results[0]['VMAFOSSEXEC_motion2_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_adm2_score'], results[0]['VMAFOSSEXEC_adm2_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_float_psnr_score'], results[0]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_float_ssim_score'], results[0]['VMAFOSSEXEC_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_float_ms_ssim_score'], results[0]['VMAFOSSEXEC_ms_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_psnr_y_score'], results[0]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_psnr_cb_score'], 38.4494410625, places=4)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_psnr_cr_score'], 40.99191027083334, places=4)

        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale0_score'], results[1]['VMAFOSSEXEC_vif_scale0_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale1_score'], results[1]['VMAFOSSEXEC_vif_scale1_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale2_score'], results[1]['VMAFOSSEXEC_vif_scale2_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale3_score'], results[1]['VMAFOSSEXEC_vif_scale3_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_motion2_score'], results[1]['VMAFOSSEXEC_motion2_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_adm2_score'], results[1]['VMAFOSSEXEC_adm2_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_float_psnr_score'], results[1]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_float_ssim_score'], results[1]['VMAFOSSEXEC_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_float_ms_ssim_score'], results[1]['VMAFOSSEXEC_ms_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_psnr_y_score'], results[1]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_psnr_cb_score'], 60.0, places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_psnr_cr_score'], 60.0, places=4)

        self.assertAlmostEqual(results_rc[0]['VMAFRC_score'], results[0]['VMAFOSSEXEC_score'], places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_score'], results[1]['VMAFOSSEXEC_score'], places=4)

    def test_run_vmafrc_compare_directly_with_ossexec_420_10bit(self):

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

        self.runner = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
                'psnr': True,
            }
        )
        self.runner.run(parallelize=True)

        results_rc = self.runner.results

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale0_score'], results[0]['VMAFOSSEXEC_vif_scale0_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale1_score'], results[0]['VMAFOSSEXEC_vif_scale1_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale2_score'], results[0]['VMAFOSSEXEC_vif_scale2_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale3_score'], results[0]['VMAFOSSEXEC_vif_scale3_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_motion2_score'], results[0]['VMAFOSSEXEC_motion2_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_adm2_score'], results[0]['VMAFOSSEXEC_adm2_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_float_psnr_score'], results[0]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_float_ssim_score'], results[0]['VMAFOSSEXEC_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_float_ms_ssim_score'], results[0]['VMAFOSSEXEC_ms_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_psnr_y_score'], results[0]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_psnr_cb_score'], 50.2879106, places=4)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_psnr_cr_score'], 49.740259, places=4)

        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale0_score'], results[1]['VMAFOSSEXEC_vif_scale0_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale1_score'], results[1]['VMAFOSSEXEC_vif_scale1_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale2_score'], results[1]['VMAFOSSEXEC_vif_scale2_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale3_score'], results[1]['VMAFOSSEXEC_vif_scale3_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_motion2_score'], results[1]['VMAFOSSEXEC_motion2_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_adm2_score'], results[1]['VMAFOSSEXEC_adm2_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_float_psnr_score'], results[1]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_float_ssim_score'], results[1]['VMAFOSSEXEC_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_float_ms_ssim_score'], results[1]['VMAFOSSEXEC_ms_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_psnr_y_score'], results[1]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_psnr_cb_score'], 72.0, places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_psnr_cr_score'], 72.0, places=4)

        self.assertAlmostEqual(results_rc[0]['VMAFRC_score'], results[0]['VMAFOSSEXEC_score'], places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_score'], results[1]['VMAFOSSEXEC_score'], places=4)

    def test_run_vmafrc_compare_directly_with_ossexec_422_10bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing()

        self.runner = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'float_psnr': True,
                'float_ssim': True,
                'float_ms_ssim': True,
                'psnr': True,
            }
        )
        self.runner.run(parallelize=True)

        results_rc = self.runner.results

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale0_score'], results[0]['VMAFOSSEXEC_vif_scale0_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale1_score'], results[0]['VMAFOSSEXEC_vif_scale1_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale2_score'], results[0]['VMAFOSSEXEC_vif_scale2_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_vif_scale3_score'], results[0]['VMAFOSSEXEC_vif_scale3_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_motion2_score'], results[0]['VMAFOSSEXEC_motion2_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_adm2_score'], results[0]['VMAFOSSEXEC_adm2_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_float_psnr_score'], results[0]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_float_ssim_score'], results[0]['VMAFOSSEXEC_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_float_ms_ssim_score'], results[0]['VMAFOSSEXEC_ms_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_psnr_y_score'], results[0]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_psnr_cb_score'], 38.7698320625, places=4)
        self.assertAlmostEqual(results_rc[0]['VMAFRC_psnr_cr_score'], 41.284188416666666, places=4)

        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale0_score'], results[1]['VMAFOSSEXEC_vif_scale0_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale1_score'], results[1]['VMAFOSSEXEC_vif_scale1_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale2_score'], results[1]['VMAFOSSEXEC_vif_scale2_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_vif_scale3_score'], results[1]['VMAFOSSEXEC_vif_scale3_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_motion2_score'], results[1]['VMAFOSSEXEC_motion2_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_adm2_score'], results[1]['VMAFOSSEXEC_adm2_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_float_psnr_score'], results[1]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_float_ssim_score'], results[1]['VMAFOSSEXEC_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_float_ms_ssim_score'], results[1]['VMAFOSSEXEC_ms_ssim_score'], places=5)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_psnr_y_score'], results[1]['VMAFOSSEXEC_psnr_score'], places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_psnr_cb_score'], 72.0, places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_psnr_cr_score'], 72.0, places=4)

        self.assertAlmostEqual(results_rc[0]['VMAFRC_score'], results[0]['VMAFOSSEXEC_score'], places=4)
        self.assertAlmostEqual(results_rc[1]['VMAFRC_score'], results[1]['VMAFOSSEXEC_score'], places=4)

    def test_run_vmafrc_runner_float_moment(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_float_moment_ref1st_score'], 59.788567354166666, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_moment_dis1st_score'], 61.332006625, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_moment_ref2nd_score'], 4696.668388125, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_moment_dis2nd_score'], 4798.659574041666, places=4)

    def test_run_vmafrc_runner_akiyo_multiply(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        try: self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 1.116691, places=4)  # 1.116691484215469
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 1.052403, places=4)  # 1.0522544319369052
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 1.070149, places=4)  # 1.0705609423182443
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 1.072518, places=4)  # 1.0731529493098957
        except AssertionError as e: self.verificationErrors.append(str(e))
        try: self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 1.072512, places=4)  # 1.0728060231246508
        except AssertionError as e: self.verificationErrors.append(str(e))

        try: self.assertAlmostEqual(results[0]['VMAFRC_score'], 132.732952, places=4)  # 132.78849246495625
        except AssertionError as e: self.verificationErrors.append(str(e))

    def test_run_vmafrc_runner_akiyo_multiply_with_feature_enhn_gain_limit(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'adm_enhn_gain_limit': 1.0, 'vif_enhn_gain_limit': 1.0}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 88.030463, places=4)  # 132.78849246495625

    def test_run_vmafrc_runner_akiyo_multiply_with_feature_enhn_gain_limit_custom(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'adm_enhn_gain_limit': 1.2, 'vif_enhn_gain_limit': 1.1}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 1.116595, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 1.029765, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 1.046767, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 1.049025, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 1.0491232394147363, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 129.474226, places=4)  # 132.78849246495625

    def test_run_vmafrc_runner_akiyo_multiply_disable_enhn_gain(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'disable_enhn_gain': True}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 88.030463, places=4)  # 132.78849246495625

    def test_run_vmafrc_runner_akiyo_multiply_no_enhn_gain_model(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1neg.json")}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 88.032956, places=4)  # 132.78849246495625

    def test_run_vmafrc_runner_akiyo_multiply_no_enhn_gain_model_and_cmd_options(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1neg.json"),
                           'adm_enhn_gain_limit': 1.2}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 1.116595, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 122.804272, places=4)  # 132.78849246495625

    def test_run_vmafrc_runner_with_enhn_gain_enabled_disabled(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'models': [
                    'path={}:name=vmaf'.format(VmafConfig.model_path("vmaf_float_v0.6.1.json")),
                    'path={}:name=vmafneg'.format(VmafConfig.model_path("vmaf_float_v0.6.1neg.json")),
                ]
            }
        )
        with self.assertRaises(AssertionError, msg="vmaf_float_v0.6.1.json and vmaf_float_v0.6.1neg.json require the same fex with "
                                                   "different input arguments, but the exception is not raised."):
            self.runner.run(parallelize=False)

    def test_run_vmafrc_runner_with_enhn_gain_enabled_disabled2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'models': [
                    'path={}:name=vmafneg'.format(VmafConfig.model_path("vmaf_float_v0.6.1neg.json")),
                    'path={}:name=vmaf'.format(VmafConfig.model_path("vmaf_float_v0.6.1.json")),
                ]
            }
        )
        with self.assertRaises(AssertionError, msg="vmaf_float_v0.6.1neg.json and vmaf_float_v0.6.1.json require different input "
                                                   "arguments for the same fex, but the exception is not raised."):
            self.runner.run(parallelize=False)

    def test_run_vmafrc_runner_akiyo_multiply_no_enhn_gain_model_inconsist(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True,
                           'model_filepath': VmafConfig.test_resource_path("vmaf_float_v0.6.1neg.inconsist.json")}
        )
        with self.assertRaises(AssertionError,
                               msg="vmaf_float_v0.6.1neg.inconsist.json has inconsistent vif_enhn_gain_limit "
                                   "values for the same fex but the exception is not raised."):
            self.runner.run(parallelize=False)

    def test_run_vmafrc_runner_akiyo_multiply_no_enhn_gain_model_and_cmd_options_illegal(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
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

    def test_run_vmafrc_runner_akiyo_multiply_no_enhn_gain_model_and_cmd_options_illegal2(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
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

    def test_run_vmafrc_runner_akiyo_multiply_with_feature_enhn_gain_limit_b_v063(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 88.4895, places=4)  # 88.032956

    def test_run_vmafrc_runner_use_default_built_in_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345149030293786, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'],0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.946416604585025, places=4)


class VmafrcQualityRunnerSubsamplingTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner0'):
            self.runner0.remove_results()
        if hasattr(self, 'runner'):
            self.runner.remove_results()

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_vmafrc_runner_with_subsample2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        subsample = 5

        self.runner0 = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={}
        )
        self.runner0.run()
        results0 = self.runner0.results

        self.runner = VmafrcQualityRunner(
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
                self.assertAlmostEqual(results0[0]['VMAFRC_scores'][i], results[0]['VMAFRC_scores'][i // subsample], places=7)
                self.assertAlmostEqual(results0[1]['VMAFRC_scores'][i], results[1]['VMAFRC_scores'][i // subsample], places=7)


class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmafrc_quality_runner_version(self):
        self.assertEqual(VmafrcQualityRunner.VERSION, 'F0.2.7-0.6.1')
        self.assertEqual(VmafrcQualityRunner.ALGO_VERSION, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
