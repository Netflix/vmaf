from __future__ import absolute_import

import os
import unittest
import re

from vmaf.config import VmafConfig
from vmaf.core.feature_extractor import VmafFeatureExtractor, MomentFeatureExtractor, \
    PsnrFeatureExtractor, SsimFeatureExtractor, MsSsimFeatureExtractor, VifFrameDifferenceFeatureExtractor
from vmaf.core.asset import Asset
from vmaf.core.result_store import FileSystemResultStore

from .testutil import set_default_576_324_videos_for_testing, set_default_flat_1920_1080_videos_for_testing

__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class FeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480})
        fextractor = VmafFeatureExtractor([asset], None)
        self.assertEquals(fextractor.executor_id, "VMAF_feature_V0.2.4c")

    def test_get_log_file_path(self):
        import hashlib

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,},
                      workdir_root="my_workdir_root")

        fextractor = VmafFeatureExtractor([asset], None)
        log_file_path = fextractor._get_log_file_path(asset)
        h = hashlib.sha1("test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480".encode("utf-8")).hexdigest()
        self.assertTrue(re.match(r"^my_workdir_root/[a-zA-Z0-9-]+/VMAF_feature_V0.2.4c_{}$".format(h), log_file_path))

    def test_run_vmaf_fextractor(self):
        print('test on running VMAF feature extractor...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'],0.93458780728708746, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.93458780728708746, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 712650.023478, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597314.95249, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 371.83541406249998, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.89395660507453423, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.93010045185439161, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.96503534602850938, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.72722361912801026, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_score'], 0.92425293429958566, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1597314.86733, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1597314.95249, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.9266444375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm3_score'], 1.0, places=4)

    def test_run_vif_frame_difference_fextractor(self):
        print('test on running VIF frame difference feature extractor...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VifFrameDifferenceFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VifDiff_feature_vifdiff_score'], 0.26745858333333333, places=4)

        self.assertAlmostEqual(results[0]['VifDiff_feature_vifdiff_num_score'], 305412.7661844375, places=0)
        self.assertAlmostEqual(results[0]['VifDiff_feature_vifdiff_den_score'], 1113927.6002349583, places=0)

        self.assertAlmostEqual(results[1]['VifDiff_feature_vifdiff_score'], 0.9791655833333334, places=4)

        self.assertAlmostEqual(results[1]['VifDiff_feature_vifdiff_num_score'], 1113926.2941030415, places=0)
        self.assertAlmostEqual(results[1]['VifDiff_feature_vifdiff_den_score'], 1113927.6002349583, places=0)

    def test_run_vmaf_fextractor_with_result_store(self):
        print('test on running VMAF feature extractor with result store...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        result_store = FileSystemResultStore(logger=None)

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=result_store
        )

        print('    running for the first time with fresh calculation...')
        self.fextractor.run()
        result0, result1 = self.fextractor.results

        self.assertTrue(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertTrue(os.path.exists(result_store._get_result_file_path(result1)))

        print('    running for the second time with stored results...')
        self.fextractor.run()
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.93458780728708746, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 712650.023478, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597314.95249, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 371.83541406249998, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.89395660507453423, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.93010045185439161, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.96503534602850938, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.72722361912801026, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_score'], 0.92425293429958566, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1597314.86733, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1597314.95249, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.9266444375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm3_score'], 1.0, places=4)

    def test_run_moment_fextractor(self):
        print('test on running Moment feature extractor...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

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
        print('test on running PSNR feature extractor...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = PsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PSNR_feature_psnr_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_feature_psnr_score'], 60.0, places=4)

    def test_run_ssim_fextractor(self):
        print('test on running SSIM feature extractor...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = SsimFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_l_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_c_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_s_score'], 0.89773633333333336, places=4)

        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_l_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_c_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_s_score'], 1.0, places=4)

    def test_run_ms_ssim_fextractor(self):
        print('test on running MS-SSIM feature extractor...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = MsSsimFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_score'], 0.9632498125, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale0_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale0_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale0_score'], 0.89773633333333336, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale1_score'], 0.99899612500000001, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale1_score'], 0.9857694375, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale1_score'], 0.941185875, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale2_score'], 0.99923564583333324, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale2_score'], 0.997034020833, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale2_score'], 0.977992145833, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale3_score'], 0.99929210416666658, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale3_score'], 0.999588104167, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale3_score'], 0.99387125, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale4_score'], 0.99940356249999995, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale4_score'], 0.999907625, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale4_score'], 0.998222583333, places=4)

        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale4_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale4_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale4_score'], 1., places=4)

    def test_run_vmaf_fextractor_checkerboard(self):
        print('test on running VMAF feature extractor on checkerboard...')

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        dis_path2 = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_1_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':1920, 'height':1080})

        asset2 = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path2,
                      asset_dict={'width':1920, 'height':1080})

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original, asset2],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.053996333333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.053996333333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.018034879735107798, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 12.554836666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 12.554836666666667, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 12.554836666666667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 12.554836666666667, places=4)

        self.assertAlmostEqual(results[2]['VMAF_feature_adm_score'], 0.78533833333333336, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_adm2_score'], 0.7853384465157921, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_adm_scale0_score'], 0.72132189911792899, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_adm_scale1_score'], 0.69259738857522501, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_adm_scale2_score'], 0.80415911639244586, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_adm_scale3_score'], 0.82791889676239039, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_motion_score'], 12.554836666666667, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_motion2_score'], 12.554836666666667, places=4)

    def test_run_vmaf_fextractor_flat(self):
        print('test on running VMAF feature extractor on flat pattern...')
        ref_path, dis_path, asset, asset_original = set_default_flat_1920_1080_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale3_score'], 1.0, places=4)


class ParallelFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractors'):
            for fextractor in self.fextractors:
                fextractor.remove_results()
            pass

    def test_run_parallel_vmaf_fextractor(self):
        print('test on running VMAF feature extractor in parallel...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original, asset],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.93458780728708746, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 712650.023478, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597314.95249, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 371.83541406249998, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.89395660507453423, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.93010045185439161, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.96503534602850938, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.72722361912801026, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_score'], 0.92425293429958566, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1597314.86733, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1597314.95249, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.9266444375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[2]['VMAF_feature_vif_num_score'], 712650.023478, places=0)

    def test_run_parallel_vmaf_fextractor_with_result_store(self):
        print('test on running VMAF feature extractor with result store in parallel...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        result_store = FileSystemResultStore(logger=None)

        print('    running for the first time with fresh calculation...')
        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=result_store
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        result0, result1 = results

        self.assertTrue(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertTrue(os.path.exists(result_store._get_result_file_path(result1)))

        print('    running for the second time with stored results...')
        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original, asset],
            None, fifo_mode=True,
            result_store=result_store
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.93458780728708746, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 712650.023478, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597314.95249, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 371.83541406249998, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.89395660507453423, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.93010045185439161, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.96503534602850938, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.72722361912801026, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_score'], 0.92425293429958566, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1597314.86733, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1597314.95249, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.9266444375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[2]['VMAF_feature_vif_score'], 0.4460930625, places=4)

    def test_run_parallel_moment_fextractor(self):
        print('test on running Moment feature extractor in parallel...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = MomentFeatureExtractor(
            [asset, asset_original, asset],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
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

        self.assertAlmostEqual(results[2]['Moment_feature_ref1st_score'], 59.788567297525134, places=4)

    def test_run_parallel_ssim_fextractor(self):
        print('test on running SSIM feature extractor in parallel...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = SsimFeatureExtractor(
            [asset, asset_original, asset],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_l_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_c_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_s_score'], 0.89773633333333336, places=4)

        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_l_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_c_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_s_score'], 1.0, places=4)

        self.assertAlmostEqual(results[2]['SSIM_feature_ssim_score'], 0.86322654166666657, places=4)

    def test_run_parallel_ms_ssim_fextractor(self):
        print('test on running MS-SSIM feature extractor in parallel...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = MsSsimFeatureExtractor(
            [asset, asset_original, asset],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_score'], 0.9632498125, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale0_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale0_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale0_score'], 0.89773633333333336, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale1_score'], 0.99899612500000001, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale1_score'], 0.9857694375, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale1_score'], 0.941185875, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale2_score'], 0.99923564583333324, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale2_score'], 0.997034020833, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale2_score'], 0.977992145833, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale3_score'], 0.99929210416666658, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale3_score'], 0.999588104167, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale3_score'], 0.99387125, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_l_scale4_score'], 0.99940356249999995, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_c_scale4_score'], 0.999907625, places=4)
        self.assertAlmostEqual(results[0]['MS_SSIM_feature_ms_ssim_s_scale4_score'], 0.998222583333, places=4)

        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale0_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale1_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale2_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale3_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_l_scale4_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_c_scale4_score'], 1., places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_feature_ms_ssim_s_scale4_score'], 1., places=4)

if __name__ == '__main__':
    unittest.main()
