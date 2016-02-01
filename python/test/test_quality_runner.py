__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import unittest

from python.asset import Asset
from python.quality_runner import VmafQualityRunner
from python.config import PYTHON_ROOT
import numpy as np

class QualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'): self.runner.remove_logs()
        pass

    def test_get_log_file_path(self):

        asset = Asset(dataset="test", ref_path="dir/refvideo.yuv",
                      dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})

        runner = VmafQualityRunner([asset], None,
                                   log_file_dir="log_file_dir")
        log_file_path = runner._get_log_file_path(asset)
        expected_log_file_path = "log_file_dir/VMAF/test_refvideo_720x480_2to2" \
                                 "_vs_disvideo_720x480_2to2_q_720x480"
        self.assertEquals(log_file_path, expected_log_file_path)

    def test_run_vamf_runner_full(self):
        ref_path = PYTHON_ROOT + "/../resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = PYTHON_ROOT + "/../resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test",
                      workdir_root="workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':384, 'quality_height':216})

        self.runner = VmafQualityRunner([asset], None, pipe_mode=False,
                                   log_file_dir="workspace/log_file_dir")
        self.runner.run()

        results = self.runner.results

        # self.assertEqual(results[0]['VMAF_study_score'], 86.340347579075001)
        self.assertEqual(np.mean(results[0]['vif_score']), 0.50052374074074057)
        self.assertEqual(results[0]['motion_score'], 47.291538425925928)
        self.assertEqual(results[0]['adm_score'], 0.87104209259259258)
        self.assertEqual(results[0]['ansnr_score'], 27.035377953703701)

if __name__ == '__main__':
    unittest.main()
