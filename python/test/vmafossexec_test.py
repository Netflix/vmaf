from __future__ import absolute_import

import unittest

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from vmaf.tools.misc import MyTestCase

from test.testutil import set_default_576_324_videos_for_testing, set_default_576_324_10bit_videos_for_testing, \
    set_default_576_324_12bit_videos_for_testing, set_default_576_324_16bit_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing_b

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class VmafossexecQualityRunnerTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass
        super().tearDown()

    def test_run_vmafossexec_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)

    def test_run_vmafossexec_runner_float_ex(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1.json")}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.3634208125, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.7666474166666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.8628533333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.9159719583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.68425208333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.94641666666666, places=4)

    def test_run_vmafossexec_runner_with_thread(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'thread': 3}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_subsample(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'subsample': 5}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.9294514, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.7428478, places=4)

    def test_run_vmafossexec_runner_with_phone_score(self):

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
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 92.52240502083333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 100.0, places=4)

    def test_run_vmafossexec_runner_norm_type_none(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath':VmafConfig.model_path("other_models", "nflxtrain_norm_type_none.json"),
            },
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9345148541666667, places=4)
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

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 74.25323125, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 77.996338095161946, places=4)

    def test_run_vmafossexec_runner_yuv422p10le(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.3636620625, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.7674953125, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.8631078125, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.9157200833333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.780577083333331, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.66890489583334, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.94641666666666, places=4)

    def test_run_vmafossexec_runner_yuv420p10le_b(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing_b()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.4330893333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.830613, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.9072123333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.945896, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 32.57143333333333, places=4)  # pypsnr: 32.57145231892744
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.8978630000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9747490000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 82.56523033333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.142826, places=4)

    def test_run_vmafossexec_runner_yuv420p12le(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.4330893333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.830613, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.9072123333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.945896, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 32.57783333333334, places=4)  # pypsnr: 32.577817940053734
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.8978630000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9747490000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 84.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 82.56523033333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.142826, places=4)

    def test_run_vmafossexec_runner_yuv420p16le(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.4330893333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.830613, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.9072123333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.945896, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9517763333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 32.579806000000005, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.8978630000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9747490000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.9999998541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.9999996041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.9999991458333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 2.8104600000000004, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 108.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 82.56523033333333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.142826, places=4)

    def test_run_vmafossexec_runner_yuv420p10le_sparks(self):

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

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.9240746, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.9968371999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.9987575999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.9993221999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 0.7523685999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9981770000000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 48.81622, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.99566, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9993778000000001, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.9999990000000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.9999990000000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.9999990000000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 0.7523685999999999, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 97.90069380000001, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 98.47175940000001, places=4)

    def test_run_vmafossexec_runner_with_transform_score(self):

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
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40.json"),
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

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
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_outltein.json"),
                'enable_transform_score': True,
                'disable_clip_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

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
                'model_filepath':VmafConfig.test_resource_path("test_model_transform_add40.json"),
                'enable_transform_score':False,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

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

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafossExecQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.053996566666666669, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 14.982747333333334, places=4)

    def test_run_vmafossexec_runner_with_phone_model(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'phone_model': True,
            },
            result_store=self.result_store,
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.053996566666666669, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 14.982747333333334, places=4)

    def test_run_vmafossexec_runner_disable_avx_precise(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_avx': True}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.9157200890843669, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.66890519623612, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_motion2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath':VmafConfig.test_resource_path("test_motion2.json")
            },
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953522916666672, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953522916666672, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 78.51201666666667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 97.089554166666673, places=4)

    def test_run_vmafossexec_runner_with_ci(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_rb_v0.6.3", "vmaf_rb_v0.6.3.json"),
                'ci': True
            },
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 75.42800833333332, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.958047916666672, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bagging_score'], 73.0913125, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_bagging_score'], 99.79000416666668, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_stddev_score'], 1.1982760416666667, places=3)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_stddev_score'], 1.3028828125, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_ci_p95_lo_score'], 70.81472708333332, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_ci_p95_lo_score'], 94.79667083333334, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_ci_p95_hi_score'], 74.83768541666666, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_ci_p95_hi_score'], 99.99736666666666, places=4)

        # per model score checks
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_0001_score'], 73.25864166666666, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_0002_score'], 70.37462916666668, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_0003_score'], 71.58177291666668, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_0020_score'], 73.14455, places=3)

    def test_run_vmafossexec_runner_with_ci_and_custom_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.json'),
                'ci': True
            },
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 75.42800833333332, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.958047916666672, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bagging_score'], 75.11496458333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_bagging_score'], 99.96504855577571, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_stddev_score'], 0.6812993325967104, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_stddev_score'], 0.03947607207290399, places=4)

    def test_run_vmafossexec_runner_with_ci_and_phone_model(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_rb_v0.6.3", "vmaf_rb_v0.6.3.json"),
                'phone_model': True,
                'ci': True,
            },
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 91.71303333333334, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_bagging_score'], 90.12363958333333, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_bagging_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_stddev_score'], 0.8371132083333332, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vmaf_stddev_score'], 0.0, places=4)

        # per model score checks
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_0001_score'], 90.24346041666666, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_0002_score'], 88.17759791666667, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_0003_score'], 89.04172708333333, places=3)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vmaf_0020_score'], 90.1585875, places=3)

    def test_run_vmafossexec_runner_vmafplus1080model(self):

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                               workdir_root=VmafConfig.workdir_path(),
                               ref_path=ref_path,
                               dis_path=ref_path,
                               asset_dict={'width': 576, 'height': 324,
                                           })

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('vmafplus_laptop_480source_vmafplusv2_sa.json')
            },
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.36342048943884936, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.76664754213485187, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.86285466690193247, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.91597177803640772, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.90791927083333324, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.9649663541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.755066666666664, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.96324981249999997, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 35.4220125, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 65.559652083333333, places=4)

    def test_run_vmafossexec_runner_akiyo_multiply_no_enhn_gain_model_float(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1neg.json")}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_egl_1_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_egl_1_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_egl_1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_egl_1_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_egl_1_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 88.032956, places=2)  # 132.78849246495625

    def test_run_vmafossexec_runner_akiyo_multiply_no_enhn_gain_model_json(self):
        ref_path = VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        dis_path = VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 352, 'height': 288})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_clip_score': True, 'model_filepath': VmafConfig.model_path("vmaf_v0.6.1neg.json")}
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_egl_1_score'], 0.9574308606115118, places=4)  # 1.116691484215469
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_egl_1_score'], 0.983699512450884, places=4)  # 1.0522544319369052
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_egl_1_score'], 0.9974276726830457, places=4)  # 1.0705609423182443
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_egl_1_score'], 0.9984692380091739, places=4)  # 1.0731529493098957
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_egl_1_score'], 0.999146211879154, places=4)  # 1.0728060231246508

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 88.030463, places=4)  # 132.78849246495625

    def test_run_parallel_vmafossexec_runner_with_repeated_assets(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.66890489583334, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416666666664, places=4)
        self.assertAlmostEqual(results[2]['VMAFOSSEXEC_score'], 76.66890489583334, places=3)
        self.assertAlmostEqual(results[3]['VMAFOSSEXEC_score'], 76.66890489583334, places=3)

    def test_run_vmafossexec_runner_4k(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path("vmaf_4k_v0.6.1.json")}
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.3636620625, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.7674953125, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.8631078125, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.9157200833333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.895352291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632406874999999, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 84.95064735416668, places=4)


class VmafossexecQualityRunnerSubsamplingTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runner0'):
            self.runner0.remove_results()
        if hasattr(self, 'runner'):
            self.runner.remove_results()
        super().tearDown()

    def test_run_vmafossexec_runner_with_subsample2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        subsample = 5

        self.runner0 = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={}
        )
        self.runner0.run(parallelize=True)
        results0 = self.runner0.results

        self.runner = VmafossExecQualityRunner(
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
                self.assertAlmostEqual(results0[0]['VMAFOSSEXEC_scores'][i], results[0]['VMAFOSSEXEC_scores'][i // subsample], places=7)
                self.assertAlmostEqual(results0[1]['VMAFOSSEXEC_scores'][i], results[1]['VMAFOSSEXEC_scores'][i // subsample], places=7)


class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmafossexec_quality_runner_version(self):
        self.assertEqual(VmafossExecQualityRunner.VERSION, 'F0.2.7-0.6.1')
        self.assertEqual(VmafossExecQualityRunner.ALGO_VERSION, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
