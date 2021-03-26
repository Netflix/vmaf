from __future__ import absolute_import

import unittest
import pprint

from vmaf.core.matlab_quality_runner import SpEEDMatlabQualityRunner, \
    StrredQualityRunner, ICIDQualityRunner, SpatioTemporalVmafQualityRunner
from vmaf.config import VmafConfig, VmafExternalConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import PsnrQualityRunner, VmafQualityRunner, VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore

from test.testutil import set_default_576_324_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

from vmaf.tools.misc import MyTestCase


@unittest.skipIf(not VmafExternalConfig.ffmpeg_path(), "ffmpeg not installed")
class QualityRunnerTest(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
        super().tearDown()

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def test_run_psnr_runner_with_notyuv(self):

        ref_path = VmafConfig.test_resource_path("mp4", "Seeking_10_288_375.mp4")
        dis_path = VmafConfig.test_resource_path("mp4", "Seeking_10_288_375.mp4")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'yuv_type': 'notyuv',
                                  'quality_width': 720, 'quality_height': 480,
                                  })
        self.runner = PsnrQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['PSNR_score'], 60.0, places=4)

    def test_run_vmaf_runner_with_notyuv(self):

        ref_path = VmafConfig.test_resource_path("mp4", "Seeking_30_480_1050.mp4")
        dis_path = VmafConfig.test_resource_path("mp4", "Seeking_10_288_375.mp4")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'yuv_type': 'notyuv',
                                  'quality_width': 360, 'quality_height': 240,
                                  })
        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1.json"),
            },
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VMAF_score'], 77.61273448644646, places=4)

    def test_run_vmaf_runner_with_notyuv_jpg(self):

        ref_path = VmafConfig.test_resource_path("test_images", "bikes.jpg")
        dis_path = VmafConfig.test_resource_path("test_images", "bikes_dis.jpg")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'yuv_type': 'notyuv',
                                  'quality_width': 512, 'quality_height': 384,
                                  'workfile_yuv_type': 'yuv444p',
                                  })
        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1.json"),
            },
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VMAF_score'], 63.255016130209064, places=4)

    def test_run_vmafossexec_runner_with_notyuv(self):

        ref_path = VmafConfig.test_resource_path("mp4", "Seeking_30_480_1050.mp4")
        dis_path = VmafConfig.test_resource_path("mp4", "Seeking_10_288_375.mp4")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'yuv_type': 'notyuv',
                                  'quality_width': 360, 'quality_height': 240,
                                  })
        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 77.59110129333334, places=3)

    def test_run_psnr_runner_with_frames(self):

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'start_frame': 2, 'end_frame': 2})
        asset2 = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'ref_start_frame': 2, 'ref_end_frame': 2,
                                  'dis_start_frame': 6, 'dis_end_frame': 6,
                                  })
        self.runner = PsnrQualityRunner(
            [asset, asset2],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=False)

        results = self.runner.results
        self.assertAlmostEqual(results[0]['PSNR_score'], 30.993823, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 19.393160, places=4)

    def test_run_psnr_runner_with_frames_proc(self):

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'start_frame': 2, 'end_frame': 2,
                                  'ref_proc_callback': 'identity',
                                  })
        asset2 = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'ref_start_frame': 2, 'ref_end_frame': 2,
                                  'dis_start_frame': 6, 'dis_end_frame': 6,
                                  'dis_proc_callback': 'identity',
                                  })
        self.runner = PsnrQualityRunner(
            [asset, asset2],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=False)

        results = self.runner.results
        self.assertAlmostEqual(results[0]['PSNR_score'], 30.993823, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 19.393160, places=4)

    def test_run_vmaf_runner_eq_cmd_gamma1d5(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        asset.asset_dict['eq_cmd'] = 'gamma=1.5'
        asset_original.asset_dict['eq_cmd'] = 'gamma=1.5'

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale0_score'], 0.34796083546749507, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale1_score'], 0.7430727630647906, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale2_score'], 0.8432128968501337, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_vif_scale3_score'], 0.9003110626539442, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_motion2_score'], 3.951617145833333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_integer_feature_adm2_score'], 0.9239987660077826, places=4)

        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale1_score'], 0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion2_score'], 3.951617145833333, places=4)
        self.assertAlmostEqual(results[1]['VMAF_integer_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_ansnr_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_integer_feature_motion_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 72.90549596147889, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)


@unittest.skipIf(not VmafExternalConfig.matlab_path(), "matlab not installed")
class MatlabQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_strrred_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = StrredQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertEqual(self.runner.VERSION, "F1.3-1.1")
        self.assertAlmostEqual(results[0]['STRRED_feature_srred_score'], 3.0166328541666663, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_trred_score'], 7.338665770833333, places=4)
        self.assertAlmostEqual(results[0]['STRRED_score'], 22.336452104611016, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_score'], 0.0, places=4)

    def test_run_speed_matlab_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = SpEEDMatlabQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)

        results = self.runner.results
        self.assertAlmostEqual(results[0]['SpEED_Matlab_feature_sspeed_4_score'], 5.155523354166667, places=4)
        self.assertAlmostEqual(results[0]['SpEED_Matlab_feature_tspeed_4_score'], 15.091642416666668, places=4)
        self.assertAlmostEqual(results[0]['SpEED_Matlab_score'], 78.4927784076698, places=4)
        self.assertAlmostEqual(results[1]['SpEED_Matlab_feature_sspeed_4_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['SpEED_Matlab_feature_tspeed_4_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['SpEED_Matlab_score'], 0.0, places=4)

    def test_run_icid_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = ICIDQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertEqual(self.runner.VERSION, "F1.0-1.0")
        self.assertAlmostEqual(results[0]['ICID_score'], 0.14382252083333333, places=4)
        self.assertAlmostEqual(results[1]['ICID_score'], 0.0, places=4)

    def test_run_stvmaf(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = SpatioTemporalVmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['STVMAF_score'], 57.708169, places=4)


@unittest.skipIf(not VmafExternalConfig.matlab_path(), "matlab not installed")
class ParallelMatlabQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_strrred_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = StrredQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertEqual(self.runner.VERSION, "F1.3-1.1")
        self.assertAlmostEqual(results[0]['STRRED_feature_srred_score'], 3.0166328541666663, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_trred_score'], 7.338665770833333, places=4)
        self.assertAlmostEqual(results[0]['STRRED_score'], 22.336452104611016, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_score'], 0.0, places=4)

    def test_run_speed_matlab_runner(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = SpEEDMatlabQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)

        results = self.runner.results
        self.assertAlmostEqual(results[0]['SpEED_Matlab_feature_sspeed_4_score'], 5.155523354166667, places=4)
        self.assertAlmostEqual(results[0]['SpEED_Matlab_feature_tspeed_4_score'], 15.091642416666668, places=4)
        self.assertAlmostEqual(results[0]['SpEED_Matlab_score'], 78.4927784076698, places=4)
        self.assertAlmostEqual(results[1]['SpEED_Matlab_feature_sspeed_4_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['SpEED_Matlab_feature_tspeed_4_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['SpEED_Matlab_score'], 0.0, places=4)


@unittest.skipIf(not VmafExternalConfig.ffmpeg_path(), "ffmpeg not installed")
class QualityRunnerTiffTest(unittest.TestCase):

    def setUp(self) -> None:
        for i in range(4):
            VmafConfig.test_resource_path('tiff', 'src01_hrc00_576x324%05d.tiff' % (i+1))
            VmafConfig.test_resource_path('tiff', 'src01_hrc01_576x324%05d.tiff' % (i+1))

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def test_run_psnr_runner_with_notyuv(self):
        ref_path = VmafConfig.test_resource_path("tiff", "src01_hrc00_576x324%05d.tiff", bypass_download=True)
        dis_path = VmafConfig.test_resource_path("tiff", "src01_hrc01_576x324%05d.tiff", bypass_download=True)
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'yuv_type': 'notyuv',
                                  'quality_width': 576, 'quality_height': 324,
                                  })
        self.runner = PsnrQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        pprint.pprint(results[0]['PSNR_scores'])
        pprint.pprint(results[0]['PSNR_score'])
        self.assertAlmostEqual(results[0]['PSNR_score'], 32.12685675, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
