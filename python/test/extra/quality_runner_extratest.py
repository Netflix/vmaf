import unittest
from vmaf.config import VmafConfig, VmafExternalConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import StrredQualityRunner, PsnrQualityRunner, \
    VmafQualityRunner, VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


# @unittest.skipIf(not VmafExternalConfig.ffmpeg_path(), "ffmpeg not installed")
class QualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_psnr_runner_with_notyuv(self):
        print 'test on running PSNR runner...'
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
        print 'test on running VMAF runner...'
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
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VMAF_score'], 77.635307276411766, places=4)

    def test_run_vmaf_runner_with_notyuv_jpg(self):
        print 'test on running VMAF runner on jpg...'
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
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VMAF_score'], 63.27798081002585, places=4)

    def test_run_vmafossexec_runner_with_notyuv(self):
        print 'test on running VMAF runner...'
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
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 77.635309333333339, places=4)

    def test_run_psnr_runner_with_frames(self):
        print 'test on running PSNR runner...'
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'start_frame':2, 'end_frame':2})
        asset2 = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'ref_start_frame':2, 'ref_end_frame':2,
                                  'dis_start_frame':6, 'dis_end_frame':6,
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


@unittest.skipIf(not VmafExternalConfig.matlab_path(), "matlab not installed")
class ParallelMatlabQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_strrred_runner(self):
        print 'test on running STRRED runner...'
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        self.runner = StrredQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)

        results = self.runner.results

        self.assertEqual(self.runner.VERSION, "F1.2-1.1")
        self.assertAlmostEqual(results[0]['STRRED_feature_srred_score'], 3.0114681041666671, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_trred_score'], 7.3039486249999994, places=4)
        self.assertAlmostEqual(results[0]['STRRED_score'], 21.995608318659482, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_score'], 0.0, places=4)


class ParallelQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()


if __name__ == '__main__':
    unittest.main()
