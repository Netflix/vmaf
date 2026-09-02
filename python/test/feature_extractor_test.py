from __future__ import absolute_import

import unittest
import re

from vmaf.config import VmafConfig
from vmaf.core.feature_extractor import VmafFeatureExtractor, \
    MomentFeatureExtractor, \
    PsnrFeatureExtractor, SsimFeatureExtractor, MsSsimFeatureExtractor, \
    VifFrameDifferenceFeatureExtractor, \
    AnsnrFeatureExtractor, PyPsnrFeatureExtractor, VmafIntegerFeatureExtractor, \
    PyPsnrMaxdb100FeatureExtractor, PypsnrFeatureExtractor
from vmaf.core.asset import Asset
from vmaf.tools.misc import MyTestCase

from test.testutil import set_default_576_324_videos_for_testing, set_default_flat_1920_1080_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing, set_default_576_324_12bit_videos_for_testing, \
    set_default_576_324_16bit_videos_for_testing, set_default_576_324_10bit_videos_for_testing_b, \
    set_default_cambi_video_for_testing_b, \
    set_default_576_324_videos_for_testing_1frames, set_default_576_324_videos_for_testing_2frames, \
    set_default_576_324_videos_for_testing_3frames, set_default_576_324_videos_for_testing_4frames

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class FeatureExtractorTest(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480})
        fextractor = VmafFeatureExtractor([asset], None)
        self.assertEqual(fextractor.executor_id, "VMAF_feature_V0.2.21")

    def test_executor_id_long_opt_dict(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480})
        fextractor = VmafFeatureExtractor([asset], None,
                                          optional_dict={"some_parameter": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"})
        self.assertEqual("VMAF_feature_V0.2.21_ab1db6ba9be857303e99cbb0ef38fc4753ef1040", fextractor.executor_id)

    def test_get_log_file_path(self):
        import hashlib

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,},
                      workdir_root="my_workdir_root")

        fextractor = VmafFeatureExtractor([asset], None)
        log_file_path = fextractor._get_log_file_path(asset)
        h = hashlib.sha1("test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480".encode("utf-8")).hexdigest()
        self.assertTrue(re.match(r"^my_workdir_root/[a-zA-Z0-9-]+/VMAF_feature_V0.2.21_{}$".format(h), log_file_path))

    def test_run_vmaf_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44641939583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0488208125, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_scores'][1], 4.214337, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_scores'][2], 4.071614, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_scores'][3], 3.825699, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.894366229166667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_scores'][0], 4.57945, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_scores'][1], 4.214337, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_scores'][2], 4.071614, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_scores'][3], 3.825699, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_score'], 3.9897714375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_aim_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345148541666667, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 713112.1822103333, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597172.416483604, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 371.80645372916666, places=3)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3636595790491415, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.7674891489570371, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.8630881475272494, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9156988075602461, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.8938705209242691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.9300123587874962, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.9649663148179196, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.7274839207734184, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_score'], 0.9539779375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0488208125, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.894366229166667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1597172.416483604, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1597172.416483604, places=0)
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

    def test_run_vmaf_fextractor_with_feature_overloads(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586,
                           'vif_kernelscale': 0.5, 'motion_force_zero': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_ks_0.5_score'], 0.39877570833333326, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_force_0_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_force_0_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_force_0_scores'][2], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_force_0_scores'][3], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_force_0_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_force_0_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_force_0_scores'][2], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_force_0_scores'][3], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9521435, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_aim_scfd_0.001586_csf_1_scf_0.002893_score'], 0.015581145833333332, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9521435, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_ks_0.5_score'], 584373.5465495, places=-1)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_ks_0.5_score'], 1465114.4054362082, places=-1)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_scfd_0.001586_csf_1_scf_0.002893_score'], 427.3585356874999, places=3)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_scfd_0.001586_csf_1_scf_0.002893_score'], 448.7978570625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_ks_0.5_score'], 0.333941, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_ks_0.5_score'], 0.6084533125, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_ks_0.5_score'], 0.7512584999999999, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_ks_0.5_score'], 0.8742467083333333, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9704823116932193, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9163563393421761, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9303921044024657, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_score'], 0.965166961957821, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.6419748821456693, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9682811458333335, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_ks_0.5_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_ks_0.5_score'], 1465114.2539468957, places=-1)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_ks_0.5_score'], 1465114.2539468957, places=-1)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_scfd_0.001586_csf_1_scf_0.002893_score'], 448.7978570625, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_scfd_0.001586_csf_1_scf_0.002893_score'], 448.7978570625, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.9266444375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_ks_0.5_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_ks_0.5_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_ks_0.5_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_ks_0.5_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)

    def test_run_vmaf_fextractor_with_adm_dlm_weight(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'adm_dlm_weight': 1.0}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44641939583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0488208125, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_scores'][1], 4.214337, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_scores'][2], 4.071614, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_scores'][3], 3.825699, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.894366229166667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_scores'][0], 4.57945, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_scores'][1], 4.214337, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_scores'][2], 4.071614, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_scores'][3], 3.825699, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_score'], 3.9897714375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_aim_dlmw_1_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_dlmw_1_score'], 0.9345148541666667, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 713112.2182617291, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597172.416483604, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_dlmw_1_score'], 371.80645372916666, places=3)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_dlmw_1_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3636595790491415, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.7674891489570371, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.8630881475272494, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9156988075602461, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_dlmw_1_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_dlmw_1_score'], 0.8938705209242691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_dlmw_1_score'], 0.9300123587874962, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_dlmw_1_score'], 0.9649663148179196, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.7274839207734184, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_dlmw_1_score'], 0.9345148541666667, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0488208125, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.894366229166667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_dlmw_1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1597172.4708455626, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1597172.4708455626, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_num_dlmw_1_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_den_dlmw_1_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_anpsnr_score'], 41.9266444375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale0_dlmw_1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale1_dlmw_1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale2_dlmw_1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_scale3_dlmw_1_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm3_dlmw_1_score'], 1.0, places=4)

    def test_run_vmaf_fextractor_adm_min_val(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'adm_min_val': 0.98}
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44641939583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0488208125, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.894366229166667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_score'], 3.9897714375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_aim_min_0.98_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_min_0.98_score'], 0.9345148541666667, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 713112.2182617291, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597172.416483604, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_min_0.98_score'], 371.80645372916666, places=3)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_min_0.98_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3636595790491415, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.7674891489570371, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.8630881475272494, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9156988075602461, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_min_0.98_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_min_0.98_score'], 0.8938705209242691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_min_0.98_score'], 0.9300123587874962, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_min_0.98_score'], 0.9649663148179196, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.7274839207734184, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_min_0.98_score'], 0.98, places=4)

    def test_run_vmaf_fextractor_adm_f1f2(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={
                'adm_f1s0': 0.017381534182398396,
                'adm_f1s1': 0.03198481448942427,
                'adm_f1s2': 0.043372664676924785,
                'adm_f1s3': 0.04567341000644211,
                'adm_f2s0': 0.005890686555214082,
                'adm_f2s1': 0.014299066724184153,
                'adm_f2s2': 0.02439691294603169,
                'adm_f2s3': 0.031312735066387855,
            }
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_f1s0_0.0173815_f1s1_0.0319848_f1s2_0.0433727_f1s3_0.0456734_f2s0_0.00589069_f2s1_0.0142991_f2s2_0.0243969_f2s3_0.0313127_score'], 0.9539779375, places=4)

    def test_run_vmaf_fextractor_with_vif_skip_scale0(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'vif_skip_scale0': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_ssclz_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_scale0_ssclz_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_scale0_ssclz_score'], -1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_ssclz_score'], 0.7674891489570371, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_ssclz_score'], 0.8630881475272494, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_ssclz_score'], 0.9156988075602461, places=4)

    def test_run_vmaf_fextractor_with_adm_skip_scale0(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'adm_skip_scale0': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9383569375000000, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_aim_ssz_score'], 0.0233171666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_ssz_score'], 0.9383569375000000, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_ssz_score'], 326.2803074791666518, places=3)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_ssz_score'], 347.6899398124999720, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_ssz_score'], 0.0000000000000000, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_ssz_score'], 0.8938705625000001, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_ssz_score'], 0.9300123749999999, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_ssz_score'], 0.9649663541666667, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_ssz_score'], 0.9575199583333333, places=4)

    def test_run_vmaf_fextractor_with_vif_sigma_nsq(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'vif_sigma_nsq': 1.5}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_snsq_1.5_score'], 0.34616335416666666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_snsq_1.5_score'], 0.7433470833333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_snsq_1.5_score'], 0.8450055416666666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_snsq_1.5_score'], 0.9031788124999999, places=4)

    def test_run_vmaf_fextractor_with_vif_scale1_min_val(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'vif_scale1_min_val': 0.987654}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_s1miv_0.987654_score'], 0.3636628333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_s1miv_0.987654_score'], 0.987654, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_s1miv_0.987654_score'], 0.8630881475272494, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_s1miv_0.987654_score'], 0.9156988075602461, places=4)

    def test_run_vmaf_fextractor_with_vif_scale2_min_val(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'vif_scale2_min_val': 0.987654}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_s2miv_0.987654_score'], 0.3636628333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_s2miv_0.987654_score'], 0.7674898958333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_s2miv_0.987654_score'], 0.987654, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_s2miv_0.987654_score'], 0.9156988075602461, places=4)

    def test_run_vmaf_fextractor_with_vif_scale3_min_val(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'vif_scale3_min_val': 0.987654}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_s3miv_0.987654_score'], 0.3636628333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_s3miv_0.987654_score'], 0.7674898958333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_s3miv_0.987654_score'], 0.8630880833333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_s3miv_0.987654_score'], 0.987654, places=4)

    def test_run_vmaf_fextractor_motion_blend(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'motion_blend_factor': 0.5, 'motion_blend_offset': 3.0}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44641939583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_mbf_0.5_mbo_3_score'], 4.0488208125, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mbf_0.5_mbo_3_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mbf_0.5_mbo_3_scores'][1], 4.214337, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mbf_0.5_mbo_3_scores'][2], 4.071614, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mbf_0.5_mbo_3_scores'][3], 3.825699, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mbf_0.5_mbo_3_score'], 3.894366229166667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mbf_0.5_mbo_3_scores'][0], 3.789725, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mbf_0.5_mbo_3_scores'][1], 3.607168, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mbf_0.5_mbo_3_scores'][2], 3.535807, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mbf_0.5_mbo_3_scores'][3], 3.412849, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mbf_0.5_mbo_3_score'], 3.494885791666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_aim_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345148541666667, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 713112.2182617291, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597172.416483604, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 371.80645372916666, places=3)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3636595790491415, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.7674891489570371, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.8630881475272494, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9156988075602461, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.8938705209242691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.9300123587874962, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.9649663148179196, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.7274839207734184, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_score'], 0.9539779375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_mbf_0.5_mbo_3_score'], 4.0488208125, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_mbf_0.5_mbo_3_score'], 3.894366229166667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1597172.4708455626, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1597172.4708455626, places=0)
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

    def test_run_vmaf_fextractor_motion_no_filter(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'motion_filter_size': 1}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44641939583333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_mfs_1_score'], 8.448365666666666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfs_1_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfs_1_scores'][1], 9.40154, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfs_1_scores'][2], 9.0379, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfs_1_scores'][3], 8.511205, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfs_1_score'], 8.1035535, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfs_1_scores'][0], 10.197686, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfs_1_scores'][1], 9.40154, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfs_1_scores'][2], 9.0379, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfs_1_scores'][3], 8.511205, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfs_1_score'], 8.316005291666666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_aim_score'], 0.026559020833333336, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345148541666667, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_num_score'], 713112.2182617291, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_den_score'], 1597172.416483604, places=0)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_num_score'], 371.80645372916666, places=3)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_den_score'], 397.83378972916671, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3636595790491415, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.7674891489570371, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.8630881475272494, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9156988075602461, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.8938705209242691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.9300123587874962, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.9649663148179196, places=4)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif2_score'], 0.7274839207734184, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm3_score'], 0.9539779375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_mfs_1_score'], 8.448365666666666, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_mfs_1_score'], 8.1035535, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_num_score'], 1597172.4708455626, places=0)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_den_score'], 1597172.4708455626, places=0)
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

    def test_run_vmaf_fextractor_motion_fps_w2d5(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'motion_fps_weight': 2.5}
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_motion_mfw_2.5_score'], 10.122052041666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfw_2.5_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfw_2.5_scores'][1], 10.535842, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfw_2.5_scores'][2], 10.179034, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfw_2.5_scores'][3], 9.564247, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mfw_2.5_score'], 9.735915666666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfw_2.5_scores'][0], 11.448624, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfw_2.5_scores'][1], 10.535842, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfw_2.5_scores'][2], 10.179034, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfw_2.5_scores'][3], 9.564247, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mfw_2.5_score'], 9.974428666666666, places=4)

    def test_run_vmaf_fextractor_motion_max_val(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'motion_max_val': 0.987}
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_motion_mmxv_0.987_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_mmxv_0.987_scores'][1], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_mmxv_0.987_scores'][2], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_mmxv_0.987_scores'][3], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_mmxv_0.987_score'], 0.9664375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mmxv_0.987_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mmxv_0.987_scores'][1], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mmxv_0.987_scores'][2], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mmxv_0.987_scores'][3], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_mmxv_0.987_score'], 0.9664375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mmxv_0.987_scores'][0], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mmxv_0.987_scores'][1], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mmxv_0.987_scores'][2], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mmxv_0.987_scores'][3], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_mmxv_0.987_score'], 0.987, places=4)

    def test_run_vmaf_fextractor_single_frame(self):

        ref_path, dis_path, asset, asset_original = set_default_cambi_video_for_testing_b()

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion3_score'], 0.0, places=4)

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
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_score'], 4.048813375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][2], 4.071609, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][3], 3.825691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 3.8943597291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][0], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][2], 4.071609, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][3], 3.825691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_score'], 3.9897647708333337, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_aim_score'], 0.026560104166666664, places=4)
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
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_score'], 0.9539728125, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_score'], 4.048813375, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 3.8943597291666667, places=4)
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

    def test_run_vmaf_integer_fextractor_dlmw(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_dlm_weight': 0.7}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_aim_dlmw_0.7_score'], 0.026560104166666664, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_dlmw_0.7_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_dlmw_0.7_score'], 0.946186, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_aim_dlmw_0.7_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_dlmw_0.7_score'], 1.000002, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm3_dlmw_0.7_score'], 1.000001, places=4)

    def test_run_vmaf_integer_fextractor_with_vif_skip_scale0(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'vif_skip_scale0': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_ssclz_score'], 0.7908157916666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_score'], 4.048813375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][2], 4.071609, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][3], 3.825691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 3.8943597291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][0], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][2], 4.071609, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][3], 3.825691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_score'], 3.9897647708333337, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_aim_score'], 0.026560104166666664, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_num_ssclz_score'], 244738.99123129167, places=0)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_den_ssclz_score'], 309454.21836345835, places=0)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_num_score'], 371.8243668541666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_den_score'], 397.8567857291667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_ssclz_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_ssclz_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_ssclz_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_ssclz_score'], 0.9157200890843669, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale0_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale1_score'], 0.8938705209242691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale2_score'], 0.9300123587874962, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale3_score'], 0.9649663148179196, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif2_score'], 0.6365807859623049, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_score'], 0.9539728125, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_ssclz_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_score'], 4.048813375, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 3.8943597291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_num_ssclz_score'], 309454.06785075, places=0)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_den_ssclz_score'], 309454.06785075, places=0)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_num_score'], 397.8576817708333, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_den_score'], 397.8567857291667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_anpsnr_score'], 41.9266444375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_ssclz_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_ssclz_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_ssclz_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_ssclz_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif2_score'], 0.7499995206917035, places=4) # is not unity, because there is a /4 in post-processing, due to assuming 4 scales
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm3_score'], 1.0, places=4)

    def test_run_vmaf_integer_fextractor_with_adm_skip_scale0(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_skip_scale0': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_ssz_score'], 0.9383448125000000, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_aim_ssz_score'], 0.0233181875000000, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_ssz_score'], 0.9383448125000000, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_num_ssz_score'], 326.2976481250000234, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_den_ssz_score'], 347.7128103541666064, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale0_ssz_score'], 0.0000000000000000, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale1_ssz_score'], 0.8938636458333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale2_ssz_score'], 0.9299980624999998, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale3_ssz_score'], 0.9649512916666666, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_ssz_score'], 0.9575132916666668, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_ssz_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_ssz_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_num_ssz_score'], 347.7129672500000197, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_den_ssz_score'], 347.7128103541666064, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale0_ssz_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale1_ssz_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale2_ssz_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale3_ssz_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm3_ssz_score'], 1.0, places=4)

    def test_run_vmaf_integer_fextractor_with_feature_overloads(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893,
                           'adm_csf_diag_scale': 0.001586, 'motion_force_zero': True}
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_score'], 0.44642331250000006, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_force_0_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_force_0_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_force_0_scores'][2], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_force_0_scores'][3], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_force_0_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_force_0_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_force_0_scores'][2], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_force_0_scores'][3], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9521308541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_aim_scfd_0.001586_csf_1_scf_0.002893_score'], 0.015581666666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9521308714259776, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_num_score'], 713111.410502125, places=0)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_den_score'], 1597165.5464884583, places=0)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_num_scfd_0.001586_csf_1_scf_0.002893_score'], 427.383783875, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_den_scfd_0.001586_csf_1_scf_0.002893_score'], 448.8302469375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9157200890843669, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9704815028776613, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9163508525033196, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9303777521891563, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9651513078629512, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif2_score'], 0.72749630372849, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_score'], 0.9682745416666666, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_force_0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_num_score'], 1597165.34910075, places=0)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_den_score'], 1597165.5464884583, places=0)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_num_scfd_0.001586_csf_1_scf_0.002893_score'], 448.8302811875001, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_den_scfd_0.001586_csf_1_scf_0.002893_score'], 448.8302469375, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_anpsnr_score'], 41.9266444375, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale0_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale1_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale2_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_scale3_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_score'], 1.0, places=4)

    def test_run_vmaf_integer_fextractor_adm_min_val(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_min_val': 0.98}
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_score'], 0.44642331250000006, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_scores'][1], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_scores'][2], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_scores'][3], 4.071609, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_score'], 4.048813375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][2], 4.071609, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][3], 3.825691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 3.8943597291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][0], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][2], 4.071609, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][3], 3.825691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_score'], 3.9897647708333337, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_min_0.98_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_aim_min_0.98_score'], 0.026560104166666664, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_min_0.98_score'], 0.9345148541666667, places=4) # at version 0.2.4b (ioannis adm fix), adm and adm2 are now identical
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_num_score'], 713111.410502125, places=0)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_den_score'], 1597165.5464884583, places=0)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_num_min_0.98_score'], 371.8243668541666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_den_min_0.98_score'], 397.8567857291667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_anpsnr_score'], 34.164776875, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.3636620710647402, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7674952820232231, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8631077727416296, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9157200890843669, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale0_min_0.98_score'], 0.90791933424090698, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale1_min_0.98_score'], 0.8938705209242691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale2_min_0.98_score'], 0.9300123587874962, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_scale3_min_0.98_score'], 0.9649663148179196, places=4)

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif2_score'], 0.72749630372849, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_min_0.98_score'], 0.98, places=4)

    def test_run_vmaf_integer_fextractor_motion_blend(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_blend_factor': 0.5, 'motion_blend_offset': 3.0}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_score'], 0.44642331250000006, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mbf_0.5_mbo_3_score'], 4.048813375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mbf_0.5_mbo_3_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mbf_0.5_mbo_3_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mbf_0.5_mbo_3_scores'][2], 4.071609, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mbf_0.5_mbo_3_scores'][3], 3.825691, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mbf_0.5_mbo_3_score'], 3.8943597291666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mbf_0.5_mbo_3_scores'][0], 3.789721, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mbf_0.5_mbo_3_scores'][1], 3.607162, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mbf_0.5_mbo_3_scores'][2], 3.535804, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mbf_0.5_mbo_3_scores'][3], 3.412845, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mbf_0.5_mbo_3_score'], 3.494882270833333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm_score'], 0.9345148541666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_aim_score'], 0.026560104166666664, places=4)
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
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_score'], 0.9539728125, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_mbf_0.5_mbo_3_score'], 4.048813375, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_mbf_0.5_mbo_3_score'], 3.8943597291666667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_aim_score'], 0.0, places=4)
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

    def test_run_vmaf_integer_fextractor_motion_max_val(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_max_val': 0.987}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mmxv_0.987_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mmxv_0.987_scores'][1], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mmxv_0.987_scores'][2], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mmxv_0.987_scores'][3], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mmxv_0.987_score'], 0.9664375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mmxv_0.987_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mmxv_0.987_scores'][1], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mmxv_0.987_scores'][2], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mmxv_0.987_scores'][3], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mmxv_0.987_score'], 0.9664375, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mmxv_0.987_scores'][0], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mmxv_0.987_scores'][1], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mmxv_0.987_scores'][2], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mmxv_0.987_scores'][3], 0.987, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mmxv_0.987_score'], 0.987, places=4)

    def test_run_vmaf_integer_fextractor_motion_five_frame_window(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_five_frame_window': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][3], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][4], 7.111204, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_score'], 6.7606451666666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][2], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][3], 7.111204, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][4], 6.969344, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_score'], 6.508453666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][0], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][1], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][2], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][3], 7.111204, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][4], 6.969344, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][5], 6.792864, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_score'], 6.836849375, places=4)

    def test_run_vmaf_integer_fextractor_motion_five_frame_window_moving_avg(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_five_frame_window': True, 'motion_moving_average': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][3], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][4], 7.111204, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_score'], 6.7606451666666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][2], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][3], 7.111204, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][4], 6.969344, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_score'], 6.508453666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][0], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][1], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][2], 7.659538, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][3], 7.274391, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][4], 7.040274, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][5], 6.881104, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_score'], 6.828103625, places=4)

    def test_run_vmaf_integer_fextractor_motion_five_frame_window_one_frame_input(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_1frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_five_frame_window': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_score'], 0.0, places=4)

    def test_run_vmaf_integer_fextractor_motion_five_frame_window_one_frame_input_moving_avg(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_1frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_five_frame_window': True, 'motion_moving_average': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_score'], 0.0, places=4)

    def test_run_vmaf_integer_fextractor_motion_five_frame_window_two_frame_input(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_2frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_five_frame_window': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_score'], 0.0, places=4)

    def test_run_vmaf_integer_fextractor_motion_five_frame_window_three_frame_input(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_3frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_five_frame_window': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_score'], 2.627165666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_score'], 2.627165666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][0], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][1], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_score'], 7.881497, places=4)

    def test_run_vmaf_integer_fextractor_motion_five_frame_window_three_frame_input_moving_avg(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_3frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_five_frame_window': True, 'motion_moving_average': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_score'], 2.627165666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_score'], 2.627165666666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][0], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][1], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_score'], 7.881497, places=4)

    def test_run_vmaf_integer_fextractor_motion_five_frame_window_four_frame_input(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_4frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_five_frame_window': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_scores'][3], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_score'], 3.8297687500000004, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][2], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_scores'][3], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_score'], 3.718789, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][0], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][1], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][2], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_scores'][3], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_score'], 7.659537500000001, places=4)

    def test_run_vmaf_integer_fextractor_motion_five_frame_window_four_frame_input_moving_avg(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_4frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_five_frame_window': True, 'motion_moving_average': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][2], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_scores'][3], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mffw_mma_score'], 3.8297687500000004, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][1], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][2], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_scores'][3], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mffw_mma_score'], 3.718789, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][0], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][1], 7.881497, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][2], 7.659538, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_scores'][3], 7.437578, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mffw_mma_score'], 7.7150275, places=4)

    def test_run_vmaf_integer_fextractor_single_frame(self):

        ref_path, dis_path, asset, asset_original = set_default_cambi_video_for_testing_b()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_score'], 0.0, places=4)

    def test_run_vmaf_integer_fextractor_single_frame_moving_avg(self):

        ref_path, dis_path, asset, asset_original = set_default_cambi_video_for_testing_b()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'motion_moving_average': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mma_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mma_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mma_score'], 0.0, places=4)

    def test_run_vmaf_integer_fextractor_motion_two_frame_input(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_2frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_scores'][1], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_score'], 2.289721, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][1], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 2.289721, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][0], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][1], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_score'], 4.579442, places=4)

    def test_run_vmaf_integer_fextractor_motion_two_frame_input_moving_avg(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_2frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_moving_average': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mma_scores'][1], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mma_score'], 2.289721, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mma_scores'][1], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mma_score'], 2.289721, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mma_scores'][0], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mma_scores'][1], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mma_score'], 4.579442, places=4)

    def test_run_vmaf_integer_fextractor_motion_three_frame_input(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_3frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_scores'][1], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_scores'][2], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_score'], 2.931255333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_scores'][2], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 2.8095493333333335, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][0], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_scores'][2], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_score'], 4.336030000000001, places=4)

    def test_run_vmaf_integer_fextractor_motion_three_frame_input_moving_avg(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing_3frames()
        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_moving_average': True}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mma_scores'][1], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mma_scores'][2], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mma_score'], 2.931255333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mma_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mma_scores'][1], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mma_scores'][2], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mma_score'], 2.8095493333333335, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mma_scores'][0], 4.579442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mma_scores'][1], 4.396883, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mma_scores'][2], 4.214324, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mma_score'], 4.396883, places=4)

    def test_run_vmaf_integer_fextractor_barten_csf_scale_coeffs_4d5h(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586, 'adm_norm_view_dist': 4.5}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results
        # TODO: feature overload should appear here too
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.959531106105518, places=4)  # float 0.9595439791666666
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.012044041666666666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nvd_4.5_score'], 0.9737435833333333, places=4)  # float 0.97375025

    def test_run_vmaf_integer_fextractor_barten_csf_scale_coeffs_4d5h_change_anw(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'adm_csf_mode': 1, 'adm_csf_scale': 0.002893, 'adm_csf_diag_scale': 0.001586,
                           'adm_norm_view_dist': 4.5, 'adm_noise_weight': 0.5}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results
        # TODO: feature overload should appear here too
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_nvd_4.5_score'], 0.9693740443602495, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_aim_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_nvd_4.5_score'], 0.009110708333333334, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm3_scfd_0.001586_csf_1_scf_0.002893_nw_0.5_nvd_4.5_score'], 0.9801317291666667, places=4)

    def test_run_vmaf_integer_fextractor_motion_fps_weight_2d5(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = VmafIntegerFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'motion_fps_weight': 2.5}
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion_mfw_2.5_score'], 10.122033166666666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mfw_2.5_scores'][0], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mfw_2.5_scores'][1], 10.535811, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mfw_2.5_scores'][2], 10.179021, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mfw_2.5_scores'][3], 9.564226, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_mfw_2.5_score'], 9.735899000000002, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mfw_2.5_scores'][0], 11.448605, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mfw_2.5_scores'][1], 10.535811, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mfw_2.5_scores'][2], 10.179021, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mfw_2.5_scores'][3], 9.564226, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion3_mfw_2.5_score'], 9.974411604166667, places=4)

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

        self.fextractor = PyPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnry_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnru_score'], 38.449441057158786, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnrv_score'], 40.9919102486235, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnry_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnru_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnrv_score'], 60.0, places=4)

    def test_run_pypsnr_fextractor_deprecated(self):

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

        self.fextractor = PyPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnry_score'], 30.780573260053277, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnru_score'], 38.769832063651364, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnrv_score'], 41.28418847734209, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnry_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnru_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnrv_score'], 72.0, places=4)

    def test_run_pypsnr_fextractor_10bit_b(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_10bit_videos_for_testing_b()

        self.fextractor = PyPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnry_score'], 32.57145231892744, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnru_score'], 39.03859552689696, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnrv_score'], 41.28060001337217, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnry_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnru_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnrv_score'], 72.0, places=4)

    def test_run_pypsnr_fextractor_12bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_12bit_videos_for_testing()

        self.fextractor = PyPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnry_score'], 32.577817940053734, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnru_score'], 39.044961148023255, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnrv_score'], 41.28696563449846, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnry_score'], 84.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnru_score'], 84.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnrv_score'], 84.0, places=4)

    def test_run_pypsnr_fextractor_16bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.fextractor = PyPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnry_score'], 32.579806240311484, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnru_score'], 39.046949448281005, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnrv_score'], 41.288953934756215, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnry_score'], 108.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnru_score'], 108.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnrv_score'], 108.0, places=4)

    def test_run_pypsnr_fextractor_16bit_custom_max_db(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.fextractor = PyPsnrFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None,
            optional_dict={'max_db': 100.0}
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnry_score'], 32.579806240311484, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnru_score'], 39.046949448281005, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_feature_psnrv_score'], 41.288953934756215, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnry_score'], 100.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnru_score'], 100.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_feature_psnrv_score'], 100.0, places=4)

    def test_run_pypsnr_fextractor_maxdb100_16bit(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_16bit_videos_for_testing()

        self.fextractor = PyPsnrMaxdb100FeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None,
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['PyPsnr_maxdb100_feature_psnry_score'], 32.579806240311484, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_maxdb100_feature_psnru_score'], 39.046949448281005, places=4)
        self.assertAlmostEqual(results[0]['PyPsnr_maxdb100_feature_psnrv_score'], 41.288953934756215, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_maxdb100_feature_psnry_score'], 100.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_maxdb100_feature_psnru_score'], 100.0, places=4)
        self.assertAlmostEqual(results[1]['PyPsnr_maxdb100_feature_psnrv_score'], 100.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
