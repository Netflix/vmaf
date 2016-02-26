__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import unittest
import config
from feature_extractor import VmafFeatureExtractor
from asset import Asset
from executor import run_executors_in_parallel
from result import FileSystemResultStore

class FeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_logs()
            self.fextractor.remove_results()
        pass

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={})
        fextractor = VmafFeatureExtractor([asset], None)
        self.assertEquals(fextractor.executor_id, "VMAF_feature_V0.1")

    def test_get_log_file_path(self):

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})

        fextractor = VmafFeatureExtractor([asset], None,
                                   log_file_dir="log_file_dir")
        log_file_path = fextractor._get_log_file_path(asset)
        expected_log_file_path = \
            "log_file_dir/VMAF_feature_V0.1/test_0_1_refvideo_720x480_2to2_vs_" \
            "disvideo_720x480_2to2_q_720x480"
        self.assertEquals(log_file_path, expected_log_file_path)

    def test_run_vamf_fextractor(self):
        print 'test on running VMAF feature extractor...'
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

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

    def test_run_vamf_fextractor_with_result_store(self):
        print 'test on running VMAF feature extractor with result store...'
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

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            result_store=result_store
        )

        print '    running for the first time with fresh calculation...'
        self.fextractor.run()
        result0, result1 = self.fextractor.results

        self.assertTrue(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertTrue(os.path.exists(result_store._get_result_file_path(result1)))

        print '    running for the second time with stored results...'
        self.fextractor.run()
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

    def test_run_vmaf_fextractor_with_scaling(self):
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':384, 'quality_height':216})

        self.fextractor = VmafFeatureExtractor(
            [asset], None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir")

        with self.assertRaises(AssertionError):
            self.fextractor.run()

    def test_run_vmaf_fextractor_not_unique(self):
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
            self.fextractor = VmafFeatureExtractor(
                [asset, asset_original],
                None, fifo_mode=True,
                log_file_dir=config.ROOT + "/workspace/log_file_dir")

class ParallelFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractors'):
            for fextractor in self.fextractors:
                fextractor.remove_logs()
                fextractor.remove_results()
            pass

    def test_run_parallel_vamf_fextractor(self):
        print 'test on running VMAF feature extractor in parallel...'
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

        self.fextractors, results = run_executors_in_parallel(
            VmafFeatureExtractor,
            [asset, asset_original],
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=None,
        )

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

    def test_run_parallel_vamf_fextractor_with_result_store(self):
        print 'test on running VMAF feature extractor with result store ' \
              'in parallel...'
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

        print '    running for the first time with fresh calculation...'
        self.fextractors, results = run_executors_in_parallel(
            VmafFeatureExtractor,
            [asset, asset_original],
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=result_store,
        )

        result0, result1 = results

        self.assertTrue(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertTrue(os.path.exists(result_store._get_result_file_path(result1)))

        print '    running for the second time with stored results...'
        _, results = run_executors_in_parallel(
            VmafFeatureExtractor,
            [asset, asset_original],
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=result_store,
        )

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

if __name__ == '__main__':
    unittest.main()
