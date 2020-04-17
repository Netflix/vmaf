from __future__ import absolute_import

import unittest

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafrcQualityRunner, VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from test.testutil import set_default_576_324_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class VmafrcQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'],0.3634208125, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7666474166666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8628533333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9159719583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345877291666667, places=4)
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

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.69926875, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.94641666666666, places=4)

    def test_run_vmafrc_runner_fixed_psnr_ssim(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'psnr': True,
                'ssim': True,
                'ms_ssim': False,  # TODO: enable fixed_ms_ssim
                'no_prediction': True,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_psnr_y_score'], 30.755063979166668, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_cb_score'], 38.4494410625, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_cr_score'], 40.99191027083334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_ssim_score'], 0.8613860416666667, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_psnr_y_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_psnr_cb_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_psnr_cr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_ssim_score'], 1.0, places=4)

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
                'ssim': True,
                'ms_ssim': False,  # TODO: enable fixed_ms_ssim
                'n_threads': 4,
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'], 0.3634208125, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7666474166666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8628533333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9159719583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345877291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9632406874999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_y_score'], 30.755063979166668, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_cb_score'], 38.4494410625, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_psnr_cr_score'], 40.99191027083334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_ssim_score'], 0.8613860416666667, places=4)

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
        self.assertAlmostEqual(results[1]['VMAFRC_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.69926875, places=4)
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
                    'path={}:name=custom_vmaf_0'.format(VmafConfig.model_path("vmaf_v0.6.1.pkl")),
                    'path={}:name=custom_vmaf_1'.format(VmafConfig.model_path("vmaf_v0.6.1.pkl")),
                ]
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_custom_vmaf_0_score'], 76.69926875, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_custom_vmaf_1_score'], 76.69926875, places=4)
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
                    'path={}:name=custom_vmaf_0'.format(VmafConfig.model_path("vmaf_v0.6.1.pkl")),
                    'path={}:name=custom_vmaf_1:enable_transform'.format(VmafConfig.model_path("vmaf_v0.6.1.pkl")),
                ]
            }
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_custom_vmaf_0_score'], 76.69926875, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_custom_vmaf_1_score'], 92.542390144364546, places=4)
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

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'],0.3634208125, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7666474166666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8628533333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9159719583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345877291666667, places=4)

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'],0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'],0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.69926875, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.94641666666666, places=4)

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

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.699266666666674, places=3)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.946416666666664, places=4)
        self.assertAlmostEqual(results[2]['VMAFRC_score'], 76.699266666666674, places=3)
        self.assertAlmostEqual(results[3]['VMAFRC_score'], 76.699266666666674, places=3)

    def test_run_vmafrc_runner_yuv422p10le(self):

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
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'],0.3634208125, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7666474166666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8628533333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9159719583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345877291666667, places=4)
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

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.69926875, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.94641666666666, places=4)

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

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 97.90585999999999, places=4)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 98.47138, places=4)

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
                'ssim': True,
                'ms_ssim': False,
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
        self.assertAlmostEqual(results_rc[0]['VMAFRC_ssim_score'], 0.8613860416666667, places=5)

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
        self.assertAlmostEqual(results_rc[1]['VMAFRC_ssim_score'], 1.0, places=5)

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
                'ssim': True,
                'ms_ssim': False,
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
        self.assertAlmostEqual(results_rc[0]['VMAFRC_ssim_score'], 0.9957616000000001, places=5)

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
        self.assertAlmostEqual(results_rc[1]['VMAFRC_ssim_score'], 1.0, places=5)

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
                'ssim': True,
                'ms_ssim': False,
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
        self.assertAlmostEqual(results_rc[0]['VMAFRC_ssim_score'], 0.8616883333333334, places=5)

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
        self.assertAlmostEqual(results_rc[1]['VMAFRC_ssim_score'], 1.0, places=5)

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
        self.runner.run()
        results = self.runner.results

        for i in range(48):
            if i % subsample == 0:
                self.assertAlmostEqual(results0[0]['VMAFRC_scores'][i], results[0]['VMAFRC_scores'][i // subsample], places=7)
                self.assertAlmostEqual(results0[1]['VMAFRC_scores'][i], results[1]['VMAFRC_scores'][i // subsample], places=7)


class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmafrc_quality_runner_version(self):
        self.assertEqual(VmafrcQualityRunner.VERSION, 'F0.2.4c-0.6.1')
        self.assertEqual(VmafrcQualityRunner.ALGO_VERSION, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
