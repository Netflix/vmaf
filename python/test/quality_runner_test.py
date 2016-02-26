__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import unittest
from asset import Asset
from quality_runner import VmafLegacyQualityRunner, VmafQualityRunner, \
    PsnrQualityRunner
from executor import run_executors_in_parallel
import config
from result import FileSystemResultStore

class QualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_logs()
            self.runner.remove_results()
        pass

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={})
        runner = VmafLegacyQualityRunner([asset], None)
        self.assertEquals(runner.executor_id, 'VMAF_legacy_V1.0')

    def test_run_vamf_legacy_runner(self):
        print 'test on running VMAF (legacy) runner...'
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

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 60.268970069698035)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 95.65756240092573)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

    def test_run_vamf_legacy_runner_with_result_store(self):
        print 'test on running VMAF (legacy) runner with result store...'
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

        result_store = FileSystemResultStore(logger=None)

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            delete_workdir=True,
            result_store=result_store
        )

        print '    running for the first time with fresh calculation...'
        self.runner.run()
        result0, result1 = self.runner.results

        # NOTE: since stored results are actually VMAF_feature's not VMAF's,
        # the two paths below shouldn't exist
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result1)))

        print '    running for the second time with stored results...'
        self.runner.run()
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 60.268970069698035)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 95.65756240092573)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

    def test_run_vmaf_legacy_runner_with_scaling(self):
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':384, 'quality_height':216})

        self.runner = VmafLegacyQualityRunner(
            [asset], None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir")

        with self.assertRaises(AssertionError):
            self.runner.run()

    def test_run_vmaf_legacy_runner_not_unique(self):
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
            self.runner = VmafLegacyQualityRunner(
                [asset, asset_original],
                None, fifo_mode=True,
                log_file_dir=config.ROOT + "/workspace/log_file_dir")

    def test_run_vmaf_runner(self):
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
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 70.2921902679039)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

    def test_run_vmaf_runner_with_model(self):
        print 'test on running VMAF runner with custom input model...'
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
            None, fifo_mode=False,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath':config.ROOT + "/resource/model/nflx_vmaff_rf_v1.pkl",
            }
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 72.8888888888889)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_score'], 98.68923611111109)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

    def test_run_psnr_runner(self):
        print 'test on running PSNR runner...'
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

        self.runner = PsnrQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0)

class ParallelQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runners'):
            for runner in self.runners: runner.remove_logs()
            pass

    def test_run_parallel_vmaf_legacy_runner(self):
        print 'test on running VMAF (legacy) quality runner in parallel...'
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
            VmafLegacyQualityRunner,
            [asset, asset_original],
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=None
        )

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 60.268970069698035)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 95.65756240092573)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

    def test_run_parallel_psnr_runner(self):
        print 'test on running PSNR quality runner in parallel...'
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
            PsnrQualityRunner,
            [asset, asset_original],
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=None
        )

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0)

    def test_run_parallel_vamf_runner_with_model(self):
        print 'test on running VMAF quality runner in parallel with custom model...'
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
            parallelize=True,
            result_store=None,
            optional_dict={
                'model_filepath':config.ROOT + "/resource/model/nflx_vmaff_rf_v1.pkl",
            }
        )

        self.assertAlmostEqual(results[0]['VMAF_score'], 72.8888888888889)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_score'], 98.68923611111109)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

if __name__ == '__main__':
    unittest.main()
