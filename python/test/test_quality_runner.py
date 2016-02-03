__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
from python.quality_runner import quality_runner_macro
from python.tools import get_stdout_logger, close_logger
from python.asset import Asset
from python.vmaf_quality_runner import VmafQualityRunner
from python import config
import numpy as np

class QualityRunnerTest(unittest.TestCase):

    def setUp(self):
        self.logger = get_stdout_logger()

    def tearDown(self):
        close_logger(self.logger)
        if hasattr(self, 'runner'): self.runner.remove_logs()

    def test_get_log_file_path(self):

        asset = Asset(dataset="test", ref_path="dir/refvideo.yuv",
                      dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})

        runner = VmafQualityRunner([asset], self.logger,
                                   log_file_dir="log_file_dir")
        log_file_path = runner._get_log_file_path(asset)
        expected_log_file_path = "log_file_dir/VMAF/test_refvideo_720x480_2to2" \
                                 "_vs_disvideo_720x480_2to2_q_720x480"
        self.assertEquals(log_file_path, expected_log_file_path)

    def test_run_vamf_runner(self):
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test",
                      workdir_root="workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_reversed = Asset(dataset="test",
                      workdir_root="workspace/workdir",
                      ref_path=dis_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        self.runner = VmafQualityRunner([asset, asset_reversed], self.logger,
                                        fifo_mode=True,
                                        log_file_dir="workspace/log_file_dir")
        self.runner.run()

        results = self.runner.results

        self.assertEqual(results[0]['VMAF_score'], 60.2689700696979)
        self.assertEqual(np.mean(results[0]['vif_score']), 0.44417014583333336)
        self.assertEqual(results[0]['motion_score'], 3.5916076041666667)
        self.assertEqual(results[0]['adm_score'], 0.91552422916666665)
        self.assertEqual(results[0]['ansnr_score'], 22.533456770833329)

        self.assertEqual(results[1]['VMAF_score'], 70.308148178177063)
        self.assertEqual(np.mean(results[1]['vif_score']), 0.48817572916666663)
        self.assertEqual(results[1]['motion_score'], 3.2422333541666659)
        self.assertEqual(results[1]['adm_score'], 0.94795422916666683)
        self.assertEqual(results[1]['ansnr_score'], 24.228765083333332)

    def test_run_vmaf_runner_with_scaling(self):
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test",
                      workdir_root="workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':384, 'quality_height':216})

        self.runner = VmafQualityRunner([asset], None, fifo_mode=False,
                                   log_file_dir="workspace/log_file_dir")

        with self.assertRaises(AssertionError):
            self.runner.run()

class ParallelQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runners'):
            for runner in self.runners:
                runner.remove_logs()

    def test_run_parallel_vamf_runner(self):
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test",
                      workdir_root="workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_reversed = Asset(dataset="test",
                      workdir_root="workspace/workdir",
                      ref_path=dis_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        self.runners, results = quality_runner_macro(
            VmafQualityRunner,
            [asset, asset_reversed],
            log_file_dir="workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True)

        self.assertEqual(results[0]['VMAF_score'], 60.2689700696979)
        self.assertEqual(np.mean(results[0]['vif_score']), 0.44417014583333336)
        self.assertEqual(results[0]['motion_score'], 3.5916076041666667)
        self.assertEqual(results[0]['adm_score'], 0.91552422916666665)
        self.assertEqual(results[0]['ansnr_score'], 22.533456770833329)

        self.assertEqual(results[1]['VMAF_score'], 70.308148178177063)
        self.assertEqual(np.mean(results[1]['vif_score']), 0.48817572916666663)
        self.assertEqual(results[1]['motion_score'], 3.2422333541666659)
        self.assertEqual(results[1]['adm_score'], 0.94795422916666683)
        self.assertEqual(results[1]['ansnr_score'], 24.228765083333332)


if __name__ == '__main__':
    unittest.main()
