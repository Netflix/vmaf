from __future__ import absolute_import

import unittest
import re

from vmaf.config import VmafConfig
from vmaf.core.feature_extractor import VmafFeatureExtractor, \
    MomentFeatureExtractor, \
    PsnrFeatureExtractor, SsimFeatureExtractor, MsSsimFeatureExtractor, \
    VifFrameDifferenceFeatureExtractor, \
    AnsnrFeatureExtractor, PypsnrFeatureExtractor, VmafIntegerFeatureExtractor, \
    PypsnrMaxdb100FeatureExtractor
from vmaf.core.asset import Asset
from vmaf.tools.misc import MyTestCase

from test.testutil import set_default_576_324_videos_for_testing, set_default_flat_1920_1080_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing, set_default_576_324_12bit_videos_for_testing, \
    set_default_576_324_16bit_videos_for_testing, set_default_576_324_10bit_videos_for_testing_b

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class FeatureExtractorTest(MyTestCase):

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480})
        fextractor = VmafFeatureExtractor([asset], None)
        self.assertEqual(fextractor.executor_id, "VMAF_feature_V0.2.7")

    def test_executor_id_long_opt_dict(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480})
        fextractor = VmafFeatureExtractor([asset], None,
                                          optional_dict={"some_parameter": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"})
        self.assertEqual("VMAF_feature_V0.2.7_ab1db6ba9be857303e99cbb0ef38fc4753ef1040", fextractor.executor_id)

    def test_get_log_file_path(self):
        import hashlib

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,},
                      workdir_root="my_workdir_root")

        fextractor = VmafFeatureExtractor([asset], None)
        log_file_path = fextractor._get_log_file_path(asset)
        h = hashlib.sha1("test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480".encode("utf-8")).hexdigest()
        self.assertTrue(re.match(r"^my_workdir_root/[a-zA-Z0-9-]+/VMAF_feature_V0.2.7_{}$".format(h), log_file_path))

    def test_run_vmaf_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345148541666667, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 712650.023478, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597314.95249, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 371.80645372916666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.8938705209242691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.9300123587874962, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.9649663148179196, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.72722361912801026, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_score'], 0.9241841443734412, places=4)

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

    def test_run_vmaf_integer_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_score'], 0.44642331250000006, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9345148541666667, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_num_score'], 713111.410502125, places=0)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_den_score'], 1597165.5464884583, places=0)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_num_score'], 371.8243668541666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_den_score'], 397.8567857291667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9157200890843669, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale1_score'], 0.8938705209242691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale2_score'], 0.9300123587874962, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale3_score'], 0.9649663148179196, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif2_score'], 0.72749630372849, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_score'], 0.9241841443734412, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_num_score'], 1597165.34910075, places=0)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_den_score'], 1597165.5464884583, places=0)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_num_score'], 397.8576817708333, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_den_score'], 397.8567857291667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_anpsnr_score'], 41.9266444375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm3_score'], 1.0, places=4)

    @unittest.skip("vifdiff alternative needed, vmaf_feature executable deprecated")
    def test_run_vif_frame_difference_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VifFrameDifferenceFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VifDiff_feature_vifdiff_score'], 0.26745858333333333, places=4)

        self.assertAlmostEqual(results[0]['VifDiff_feature_vifdiff_num_score'], 305412.7661844375, places=0)
        self.assertAlmostEqual(results[0]['VifDiff_feature_vifdiff_den_score'], 1113927.6002349583, places=0)

        self.assertAlmostEqual(results[1]['VifDiff_feature_vifdiff_score'], 0.9791655833333334, places=4)

        self.assertAlmostEqual(results[1]['VifDiff_feature_vifdiff_num_score'], 1113926.2941030415, places=0)
        self.assertAlmostEqual(results[1]['VifDiff_feature_vifdiff_den_score'], 1113927.6002349583, places=0)

    def test_run_moment_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = MomentFeatureExtractor(
            [asset, asset_original],
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

    def test_run_moment_fextractor_10bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing()

        self.fextractor = MomentFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_feature_ref1st_score'], 59.788567297525134 * 4, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_ref2nd_score'], 4696.668388042269 * 16, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_refvar_score'], 1121.519917231203 * 16, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis1st_score'], 61.332006624999984 * 4, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis2nd_score'], 4798.659574041666 * 16, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_disvar_score'], 1036.837184348847 * 16, places=4)

        self.assertAlmostEqual(results[1]['Moment_feature_ref1st_score'], 59.788567297525134 * 4, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_ref2nd_score'], 4696.668388042269 * 16, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_refvar_score'], 1121.519917231203 * 16, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis1st_score'], 59.788567297525134 * 4, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis2nd_score'], 4696.668388042269 * 16, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_disvar_score'], 1121.519917231203 * 16, places=4)

    def test_run_moment_fextractor_12bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()

        self.fextractor = MomentFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_feature_ref1st_score'], 979.6711819844536, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_ref2nd_score'], 1238135.8363054413, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_refvar_score'], 278292.25886465114, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis1st_score'], 996.2818072702333, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis2nd_score'], 1255533.4389574758, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_disvar_score'], 262952.8893540034, places=4)

        self.assertAlmostEqual(results[1]['Moment_feature_ref1st_score'], 979.6711819844536, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_ref2nd_score'], 1238135.8363054413, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_refvar_score'], 278292.25886465114, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis1st_score'], 979.6711819844536, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis2nd_score'], 1238135.8363054413, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_disvar_score'], 278292.25886465114, places=4)

    def test_run_moment_fextractor_16bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.fextractor = MomentFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_feature_ref1st_score'], 979.6711819844536 * 16.0, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_ref2nd_score'], 1238135.8363054413 * 256.0, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_refvar_score'], 278292.25886465114 * 256.0, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis1st_score'], 996.2818072702333 * 16.0, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_dis2nd_score'], 1255533.4389574758 * 256.0, places=4)
        self.assertAlmostEqual(results[0]['Moment_feature_disvar_score'], 262952.8893540034 * 256.0, places=4)

        self.assertAlmostEqual(results[1]['Moment_feature_ref1st_score'], 979.6711819844536 * 16.0, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_ref2nd_score'], 1238135.8363054413 * 256.0, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_refvar_score'], 278292.25886465114 * 256.0, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis1st_score'], 979.6711819844536 * 16.0, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_dis2nd_score'], 1238135.8363054413 * 256.0, places=4)
        self.assertAlmostEqual(results[1]['Moment_feature_disvar_score'], 278292.25886465114 * 256.0, places=4)

    def test_run_psnr_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = PsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PSNR_feature_psnr_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_feature_psnr_score'], 60.0, places=4)

    def test_run_ansnr_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = AnsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['ANSNR_feature_ansnr_score'], 23.509571520833333, places=4)
        self.assertAlmostEqual(results[0]['ANSNR_feature_anpsnr_score'], 34.16477641666666, places=4)
        self.assertAlmostEqual(results[1]['ANSNR_feature_ansnr_score'], 31.271439270833337, places=4)
        self.assertAlmostEqual(results[1]['ANSNR_feature_anpsnr_score'], 41.926644187499996, places=4)

    def test_run_ssim_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = SsimFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
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

    def test_run_ssim_fextractor_flat(self):

        ref_path, dis_path, asset, asset_original = set_default_flat_1920_1080_videos_for_testing()

        self.fextractor = SsimFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_score'], 0.9087330000000001, places=8)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_l_score'], 0.9087330000000001, places=8)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_c_score'], 1.0, places=8)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_s_score'], 1.0, places=8)

        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_score'], 1.0, places=8)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_l_score'], 1.0, places=8)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_c_score'], 1.0, places=8)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_s_score'], 1.0, places=8)

    def test_run_ms_ssim_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = MsSsimFeatureExtractor(
            [asset, asset_original],
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

    def test_run_vmaf_integer_fextractor_checkerboard(self):

        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        dis_path2 = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_1_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 1920, 'height': 1080})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 1920, 'height': 1080})

        asset2 = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path2,
                      asset_dict={'width': 1920, 'height': 1080})

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original, asset2],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_score'], 0.053996333333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.053996333333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale3_score'], 0.018034879735107798, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 12.554711666666668, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 12.554711666666668, places=4)

        self.assertAlmostEqual(results[2]['VMAF_integer_feature_adm_score'], 0.78533833333333336, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_adm2_score'], 0.7853384465157921, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_adm_scale0_score'], 0.72132189911792899, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_adm_scale1_score'], 0.69259738857522501, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_adm_scale2_score'], 0.80415911639244586, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_adm_scale3_score'], 0.82791889676239039, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_motion_score'], 12.554711666666668, places=4)
        self.assertAlmostEqual(results[2]['VMAF_integer_feature_motion2_score'], 12.554711666666668, places=4)

    def test_run_vmaf_integer_fextractor_flat(self):

        ref_path, dis_path, asset, asset_original = set_default_flat_1920_1080_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale3_score'], 1.0, places=4)

    def test_run_psnr_fextractor_proc(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        callback_dict = {
            'ref_proc_callback': 'identity',
            'dis_proc_callback': 'multiply',
        }
        asset.asset_dict.update(callback_dict)
        asset_original.asset_dict.update(callback_dict)

        self.fextractor = PsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PSNR_feature_psnr_score'], 27.645446604166665, places=8)
        self.assertAlmostEqual(results[1]['PSNR_feature_psnr_score'], 31.87683660416667, places=8)

    def test_run_pypsnr_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = PypsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnry_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnru_score'], 38.449441057158786, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnrv_score'], 40.9919102486235, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnry_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnru_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnrv_score'], 60.0, places=4)

    def test_run_pypsnr_fextractor_10bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing()

        self.fextractor = PypsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnry_score'], 30.780573260053277, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnru_score'], 38.769832063651364, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnrv_score'], 41.28418847734209, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnry_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnru_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnrv_score'], 72.0, places=4)

    def test_run_pypsnr_fextractor_10bit_b(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing_b()

        self.fextractor = PypsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnry_score'], 32.57145231892744, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnru_score'], 39.03859552689696, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnrv_score'], 41.28060001337217, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnry_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnru_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnrv_score'], 72.0, places=4)

    def test_run_pypsnr_fextractor_12bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()

        self.fextractor = PypsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnry_score'], 32.577817940053734, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnru_score'], 39.044961148023255, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnrv_score'], 41.28696563449846, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnry_score'], 84.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnru_score'], 84.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnrv_score'], 84.0, places=4)

    def test_run_pypsnr_fextractor_16bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.fextractor = PypsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnry_score'], 32.579806240311484, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnru_score'], 39.046949448281005, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnrv_score'], 41.288953934756215, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnry_score'], 108.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnru_score'], 108.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnrv_score'], 108.0, places=4)

    def test_run_pypsnr_fextractor_16bit_custom_max_db(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.fextractor = PypsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'max_db': 100.0}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnry_score'], 32.579806240311484, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnru_score'], 39.046949448281005, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_feature_psnrv_score'], 41.288953934756215, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnry_score'], 100.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnru_score'], 100.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_feature_psnrv_score'], 100.0, places=4)

    def test_run_pypsnr_fextractor_maxdb100_16bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.fextractor = PypsnrMaxdb100FeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None,
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Pypsnr_maxdb100_feature_psnry_score'], 32.579806240311484, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_maxdb100_feature_psnru_score'], 39.046949448281005, places=4)
        self.assertAlmostEqual(results[0]['Pypsnr_maxdb100_feature_psnrv_score'], 41.288953934756215, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_maxdb100_feature_psnry_score'], 100.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_maxdb100_feature_psnru_score'], 100.0, places=4)
        self.assertAlmostEqual(results[1]['Pypsnr_maxdb100_feature_psnrv_score'], 100.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
