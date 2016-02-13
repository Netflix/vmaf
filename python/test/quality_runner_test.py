__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
from asset import Asset
from vmaf_quality_runner import VmafQualityRunner
from executor import run_executors_in_parallel
import config

class QualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'): self.runner.remove_logs()
        pass

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={})
        runner = VmafQualityRunner([asset], None,
                                   log_file_dir="log_file_dir",
                                   fifo_mode=True)
        self.assertEquals(runner.executor_id, 'VMAF_V0.1')

    def test_run_vamf_runner(self):
        print 'test on running VMAF runner...'
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir")
        self.runner.run()

        results = self.runner.results

        self.assertEqual(results[0]['VMAF_score'], 60.268970069698035)
        self.assertEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertEqual(results[1]['VMAF_score'], 95.65756240092573)
        self.assertEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

    def test_run_vmaf_runner_with_scaling(self):
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':384, 'quality_height':216})

        self.runner = VmafQualityRunner(
            [asset], None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir")

        with self.assertRaises(AssertionError):
            self.runner.run()

    def test_run_vmaf_runner_not_unique(self):
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        with self.assertRaises(AssertionError):
            self.runner = VmafQualityRunner(
                [asset, asset_original],
                None, fifo_mode=True,
                log_file_dir=config.ROOT + "/workspace/log_file_dir")

class ParallelQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runners'):
            for runner in self.runners: runner.remove_logs()
            pass

    def test_run_parallel_vamf_runner(self):
        print 'test on running VMAF quality runner in parallel...'
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        self.runners, results = run_executors_in_parallel(
            VmafQualityRunner,
            [asset, asset_original],
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True)

        self.assertEqual(results[0]['VMAF_score'], 60.268970069698035)
        self.assertEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertEqual(results[1]['VMAF_score'], 95.65756240092573)
        self.assertEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

if __name__ == '__main__':
    unittest.main()
