from executor import run_executors_in_parallel

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
import config
from vmaf_feature_extractor import VmafFeatureExtractor
from asset import Asset

class FeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'): self.fextractor.remove_logs()
        pass

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={})
        fextractor = VmafFeatureExtractor([asset], None,
                                   log_file_dir="log_file_dir",
                                   fifo_mode=True)
        self.assertEquals(fextractor.executor_id, "VMAF_feature_V0.1")

    def test_get_log_file_path(self):

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})

        runner = VmafFeatureExtractor([asset], None,
                                   log_file_dir="log_file_dir",
                                   fifo_mode=True)
        log_file_path = runner._get_log_file_path(asset)
        expected_log_file_path = \
            "log_file_dir/VMAF_feature_V0.1/test_0_1_refvideo_720x480_2to2_vs_" \
            "disvideo_720x480_2to2_q_720x480"
        self.assertEquals(log_file_path, expected_log_file_path)

    def test_run_vamf_runner(self):
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
            log_file_dir=config.ROOT + "/workspace/log_file_dir")
        self.fextractor.run()

        results = self.fextractor.results

        self.assertEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

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

        self.runners, results = run_executors_in_parallel(
            VmafFeatureExtractor,
            [asset, asset_original],
            log_file_dir=config.ROOT + "/workspace/log_file_dir",
            fifo_mode=True,
            delete_workdir=True,
            parallelize=False)

        self.assertEqual(results[0]['VMAF_feature_vif_score'], 0.44417014583333336)
        self.assertEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertEqual(results[0]['VMAF_feature_adm_score'], 0.91552422916666665)
        self.assertEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329)

        self.assertEqual(results[1]['VMAF_feature_vif_score'], 1.0)
        self.assertEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667)
        self.assertEqual(results[1]['VMAF_feature_adm_score'], 1.0)
        self.assertEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322)

if __name__ == '__main__':
    unittest.main()
