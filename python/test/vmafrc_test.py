from __future__ import absolute_import

import unittest

from vmaf.core.quality_runner import VmafrcQualityRunner, VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from .testutil import set_default_576_324_videos_for_testing

__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class VmafrcQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

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

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.3634208125, places=8)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.7666474166666667, places=8)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.8628533333333334, places=8)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.9159719583333334, places=8)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.895352291666667, places=8)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.9345877291666667, places=8)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=8)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=8)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632406874999999, places=8)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=8)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.9999998541666666, places=8)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.9999996041666667, places=8)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.9999991458333334, places=8)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.895352291666667, places=8)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=8)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=8)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=8)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=8)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.69926875, places=8)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.94641666666666, places=8)

    def test_run_vmafrc_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafrcQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale0_score'],0.3634208125, places=6)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale1_score'], 0.7666474166666667, places=6)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale2_score'], 0.8628533333333334, places=5)
        self.assertAlmostEqual(results[0]['VMAFRC_vif_scale3_score'], 0.9159719583333334, places=6)
        self.assertAlmostEqual(results[0]['VMAFRC_motion2_score'], 3.895352291666667, places=6)
        self.assertAlmostEqual(results[0]['VMAFRC_adm2_score'], 0.9345877291666667, places=6)
        self.assertAlmostEqual(results[0]['VMAFRC_float_psnr_score'], 30.7550666667, places=6)
        self.assertAlmostEqual(results[0]['VMAFRC_float_ssim_score'], 0.86322654166666657, places=6)
        # self.assertAlmostEqual(results[0]['VMAFRC_float_ms_ssim_score'], 0.9632406874999999, places=4)  # TODO: add float_ms_ssim

        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale0_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale1_score'],0.9999998541666666, places=6)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale2_score'],0.9999996041666667, places=6)
        self.assertAlmostEqual(results[1]['VMAFRC_vif_scale3_score'], 0.9999991458333334, places=6)
        self.assertAlmostEqual(results[1]['VMAFRC_motion2_score'], 3.895352291666667, places=6)
        self.assertAlmostEqual(results[1]['VMAFRC_adm2_score'], 1.0, places=6)
        self.assertAlmostEqual(results[1]['VMAFRC_float_psnr_score'], 60.0, places=6)
        self.assertAlmostEqual(results[1]['VMAFRC_float_ssim_score'], 1.0, places=6)
        # self.assertAlmostEqual(results[1]['VMAFRC_float_ms_ssim_score'], 1.0, places=6)  # TODO: add float_ms_ssim

        self.assertAlmostEqual(results[0]['VMAFRC_score'], 76.69926875, places=6)
        self.assertAlmostEqual(results[1]['VMAFRC_score'], 99.94641666666666, places=6)
