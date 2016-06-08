__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import unittest
import re

import config
from core.feature_extractor import VmafFeatureExtractor, MomentFeatureExtractor, \
    PsnrFeatureExtractor
from core.asset import Asset
from core.executor import run_executors_in_parallel
from core.result_store import FileSystemResultStore


class FeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={})
        fextractor = VmafFeatureExtractor([asset], None)
        self.assertEquals(fextractor.executor_id, "VMAF_feature_V0.2.1")

    def test_get_log_file_path(self):
        import hashlib

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2},
                      workdir_root="my_workdir_root")

        fextractor = VmafFeatureExtractor([asset], None)
        log_file_path = fextractor._get_log_file_path(asset)
        h = hashlib.sha1("test_0_1_refvideo_720x480_2to2_vs_disvideo_720x480_2to2_q_720x480").hexdigest()
        self.assertTrue(re.match(r"^my_workdir_root/[a-zA-Z0-9-]+/VMAF_feature_V0.2.1_{}$".format(h), log_file_path))

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
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44455808333333313, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9254334398006141, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 644527.3311971038, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1449635.3812459996, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 6899.815530270836, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.15266368750002, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3655846219305399, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.7722301581694561, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.8681486658208089, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9207121810522212, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1449635.3522745417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1449635.3812459996, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.65012097916668, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0000000132944864, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 0.9999998271651448, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.9999998649680067, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.9999998102499, places=4)

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

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44455808333333313, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9254334398006141, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 644527.3311971038, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1449635.3812459996, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 6899.815530270836, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.15266368750002, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3655846219305399, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.7722301581694561, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.8681486658208089, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9207121810522212, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1449635.3522745417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1449635.3812459996, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.65012097916668, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0000000132944864, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 0.9999998271651448, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.9999998649680067, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.9999998102499, places=4)

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
                None, fifo_mode=True)

    def test_run_moment_fextractor(self):
        print 'test on running Moment feature extractor...'
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

        self.fextractor = MomentFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_feature_ref1st_score'], 59.788567297525134, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_ref2nd_score'], 4696.668388042269, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_refvar_score'], 1121.519917231203, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis1st_score'], 61.332006624999984, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis2nd_score'], 4798.659574041666, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_disvar_score'], 1036.837184348847, places=4)

        self.assertAlmostEqual(results[1]['Moment_feature_ref1st_score'], 59.788567297525134, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_ref2nd_score'], 4696.668388042269, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_refvar_score'], 1121.519917231203, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis1st_score'], 59.788567297525134, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis2nd_score'], 4696.668388042269, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_disvar_score'], 1121.519917231203, places=4)

    def test_run_psnr_fextractor(self):
        print 'test on running PSNR feature extractor...'
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

        self.fextractor = PsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PSNR_feature_psnr_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_feature_psnr_score'], 60.0, places=4)

class ParallelFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractors'):
            for fextractor in self.fextractors:
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
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=None,
        )

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44455808333333313, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9254334398006141, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 644527.3311971038, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1449635.3812459996, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 6899.815530270836, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.15266368750002, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3655846219305399, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.7722301581694561, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.8681486658208089, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9207121810522212, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1449635.3522745417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1449635.3812459996, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.65012097916668, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0000000132944864, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 0.9999998271651448, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.9999998649680067, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.9999998102499, places=4)

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
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=result_store,
        )

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44455808333333313, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9254334398006141, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 644527.3311971038, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1449635.3812459996, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 6899.815530270836, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.15266368750002, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3655846219305399, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.7722301581694561, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.8681486658208089, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9207121810522212, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1449635.3522745417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1449635.3812459996, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_score'], 7535.801140312499, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.65012097916668, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0000000132944864, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 0.9999998271651448, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.9999998649680067, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.9999998102499, places=4)

    def test_run_parallel_moment_fextractor(self):
        print 'test on running Moment feature extractor in parallel...'
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
            MomentFeatureExtractor,
            [asset, asset_original],
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=None,
        )

        self.assertAlmostEqual(results[0]['Moment_feature_ref1st_score'], 59.788567297525134, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_ref2nd_score'], 4696.668388042269, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_refvar_score'], 1121.519917231203, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis1st_score'], 61.332006624999984, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis2nd_score'], 4798.659574041666, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_disvar_score'], 1036.837184348847, places=4)

        self.assertAlmostEqual(results[1]['Moment_feature_ref1st_score'], 59.788567297525134, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_ref2nd_score'], 4696.668388042269, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_refvar_score'], 1121.519917231203, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis1st_score'], 59.788567297525134, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis2nd_score'], 4696.668388042269, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_disvar_score'], 1121.519917231203, places=4)


if __name__ == '__main__':
    unittest.main()
