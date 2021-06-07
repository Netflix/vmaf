from __future__ import absolute_import

import unittest

from vmaf.config import VmafConfig, VmafExternalConfig
from vmaf.core.asset import Asset
from vmaf.core.feature_extractor import VmafFeatureExtractor
from vmaf.core.matlab_feature_extractor import StrredFeatureExtractor, StrredOptFeatureExtractor, SpEEDMatlabFeatureExtractor, STMADFeatureExtractor, iCIDFeatureExtractor
from vmaf.tools.misc import MyTestCase
from vmaf.tools.stats import ListStats

from test.testutil import set_default_576_324_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


@unittest.skipIf(not VmafExternalConfig.matlab_path(), "matlab not installed")
class MatlabFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass

    def test_run_strred_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = StrredFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['STRRED_feature_srred_score'], 3.0166328541666663, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_trred_score'], 7.338665770833333, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_strred_score'], 22.336452104611016, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_strred_all_same_score'], 22.138060270044175, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_strred_score'], 0.0, places=4)

    def test_run_strredOpt_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = StrredOptFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        # notice that these numbers are the same with ST-RRED, since the opt version should always produce identical results
        self.assertAlmostEqual(results[0]['STRREDOpt_feature_srred_score'], 3.0166328541666663, places=4)
        self.assertAlmostEqual(results[0]['STRREDOpt_feature_trred_score'], 7.338665770833333, places=4)
        self.assertAlmostEqual(results[0]['STRREDOpt_feature_strred_score'], 22.336452104611016, places=4)
        self.assertAlmostEqual(results[1]['STRREDOpt_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRREDOpt_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRREDOpt_feature_strred_score'], 0.0, places=4)

    def test_run_SpEED_matlab_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = SpEEDMatlabFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )

        self.fextractor.run(parallelize=False)
        results = self.fextractor.results
        # S-SpEED assertions on first frame
        self.assertAlmostEqual(results[0].result_dict[self.fextractor.TYPE + '_sspeed_2_scores'][0], 13.510418, places=4)
        self.assertAlmostEqual(results[0].result_dict[self.fextractor.TYPE + '_sspeed_3_scores'][0], 7.211881, places=4)
        self.assertAlmostEqual(results[0].result_dict[self.fextractor.TYPE + '_sspeed_4_scores'][0], 4.921501, places=4)
        # T-SpEED assertions on third frame
        self.assertAlmostEqual(results[0].result_dict[self.fextractor.TYPE + '_tspeed_2_scores'][2], 32.994605, places=4)
        self.assertAlmostEqual(results[0].result_dict[self.fextractor.TYPE + '_tspeed_3_scores'][2], 22.404285, places=4)
        self.assertAlmostEqual(results[0].result_dict[self.fextractor.TYPE + '_tspeed_4_scores'][2], 15.233468, places=4)

    def test_run_stmad_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = STMADFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0].result_dict['STMAD_feature_smad_all_same_scores'][0], 2.889626, places=4)
        self.assertAlmostEqual(results[0].result_dict['STMAD_feature_tmad_all_same_scores'][0], 5.649214, places=4)
        self.assertAlmostEqual(results[0].result_dict['STMAD_feature_stmad_all_same_scores'][0], 4.983220, places=4)

        self.assertAlmostEqual(results[1].result_dict['STMAD_feature_smad_all_same_scores'][0], 1.000000, places=4)
        self.assertAlmostEqual(results[1].result_dict['STMAD_feature_tmad_all_same_scores'][0], 0.000000, places=4)
        self.assertAlmostEqual(results[1].result_dict['STMAD_feature_stmad_all_same_scores'][0], -1.818097, places=4)

    def test_run_iCID_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = iCIDFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['ICID_feature_icid_score'], 0.14382252083333333, places=4)
        self.assertAlmostEqual(results[1]['ICID_feature_icid_score'], 0.0, places=4)


@unittest.skipIf(not VmafExternalConfig.matlab_path(), "matlab not installed")
class ParallelMatlabFeatureExtractorTestNew(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass

    def test_run_strred_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = StrredFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['STRRED_feature_srred_score'], 3.0166328541666663, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_trred_score'], 7.338665770833333, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_strred_score'], 22.336452104611016, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_strred_all_same_score'], 22.138060270044175, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_strred_score'], 0.0, places=4)

    def test_run_strred_fextractor_blackframes(self):

        ref_path = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "flat_1920_1080_10.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324})

        from vmaf.core.result_store import FileSystemResultStore
        result_store = FileSystemResultStore(logger=None)

        self.fextractor = StrredFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=result_store
        )

        self.fextractor.run(parallelize=True)

        result0, result1 = self.fextractor.results
        import os
        self.assertTrue(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertTrue(os.path.exists(result_store._get_result_file_path(result1)))

        self.fextractor.run(parallelize=True)
        results = self.fextractor.results

        # ignore NaN
        for result in results:
            result.set_score_aggregate_method(ListStats.nonemean)

        self.assertAlmostEqual(results[0]['STRRED_feature_srred_score'], 1220.5679849999999, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_trred_score'], 50983.3097155, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_strred_score'], 62228595.6081, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_strred_score'], 0.0, places=4)


@unittest.skipIf(not VmafExternalConfig.ffmpeg_path(), "ffmpeg not installed")
class ParallelFeatureExtractorTestNew(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_run_vmaf_fextractor_with_resampling(self):
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'quality_width': 160, 'quality_height': 90})

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'quality_width': 160, 'quality_height': 90})

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=False)

        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.74165043750000004, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'],1.4066421666666666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9807496875, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 27.319241250000001, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 1.4066421666666666, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.682829895833333, places=4)

    def test_run_vmaf_fextractor_with_cropping(self):
        # crop_cmd: 288:162:144:81 - crop to 288x162 with upper-left pixel
        # starting at coordinate (144, 81)

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'crop_cmd': '288:162:144:81',
                                  'quality_width': 288, 'quality_height': 162,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'crop_cmd': '288:162:144:81',
                                  'quality_width': 288, 'quality_height': 162,
                                  })

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=False)

        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.45365762500000012, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 2.8779373333333331, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9388824973398119, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.942050354166668, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 2.8779373333333331, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.71648420833333, places=4)

    def test_run_vmaf_fextractor_with_padding(self):
        # pad_cmd: iw+100:ih+100:50:50 - pad to (iw+100)x(ih+100), where iw is
        # input width, ih is input height, and starting point is (-50, -50)

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'pad_cmd': 'iw+100:ih+100:50:50',
                                  'quality_width': 676, 'quality_height': 424,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'pad_cmd': 'iw+100:ih+100:50:50',
                                  'quality_width': 676, 'quality_height': 424,
                                  })

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=True)

        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.51023564583333325, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 2.6397702083333332, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9410537302204777, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 26.893242291666667, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 2.6397702083333332, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 34.306043416666668, places=4)

    def test_run_vmaf_fextractor_with_cropping_and_padding_to_original_wh(self):
        # crop_cmd: 288:162:144:81 - crop to the center 288x162 image
        # pad_cmd: iw+288:ih+162:144:81 - pad back to the original size

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'crop_cmd': '288:162:144:81',
                                  'pad_cmd': 'iw+288:ih+162:144:81',
                                  'quality_width': 576, 'quality_height': 324,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'crop_cmd': '288:162:144:81',
                                  'pad_cmd': 'iw+288:ih+162:144:81',
                                  'quality_width': 576, 'quality_height': 324,
                                  })

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=True)

        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.64106379166666672, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 0.7203213958333331, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9469305256822512, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 32.78451041666667, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 0.7203213958333331, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 40.280504208333333, places=4)

    def test_run_vmaf_fextractor_with_cropping_and_padding_to_original_wh_proc(self):
        # crop_cmd: 288:162:144:81 - crop to the center 288x162 image
        # pad_cmd: iw+288:ih+162:144:81 - pad back to the original size

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'crop_cmd': '288:162:144:81',
                                  'pad_cmd': 'iw+288:ih+162:144:81',
                                  'quality_width': 576, 'quality_height': 324,
                                  'ref_proc_callback': 'identity',
                                  'dis_proc_callback': 'identity',
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'crop_cmd': '288:162:144:81',
                                  'pad_cmd': 'iw+288:ih+162:144:81',
                                  'quality_width': 576, 'quality_height': 324,
                                  })

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=True)

        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.64106379166666672, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 0.7203213958333331, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9469305256822512, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 32.78451041666667, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 0.7203213958333331, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 40.280504208333333, places=4)

    def test_run_vmaf_fextractor_with_resampling_bilinear(self):
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'ref_resampling_type': 'lanczos',
                                  'dis_resampling_type': 'bilinear',
                                  'quality_width': 160, 'quality_height': 90})

        self.fextractor = VmafFeatureExtractor(
            [asset], None, fifo_mode=False)

        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.6276097500000001, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'],1.418299520833333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9412648333333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 25.377805270833335, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
