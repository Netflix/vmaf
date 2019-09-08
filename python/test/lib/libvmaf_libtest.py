import unittest
from vmaf import project_path, required
from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


def set_default_576_324_videos_for_testing():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
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

    return ref_path, dis_path, asset, asset_original


class LibRunner(VmafossExecQualityRunner):

    TYPE = "TESTLIB"

    def _get_exec(self):
        return required(project_path("wrapper/testlib"))


class QualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_testlib_runner(self):
        print('test on running TESTLIB runner...')
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
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

        self.runner = LibRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['TESTLIB_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['TESTLIB_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['TESTLIB_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['TESTLIB_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['TESTLIB_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['TESTLIB_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['TESTLIB_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['TESTLIB_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['TESTLIB_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['TESTLIB_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['TESTLIB_vif_scale1_score'], 0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['TESTLIB_vif_scale2_score'], 0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['TESTLIB_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['TESTLIB_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['TESTLIB_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['TESTLIB_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['TESTLIB_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['TESTLIB_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['TESTLIB_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['TESTLIB_score'], 99.946416604585025, places=4)

    def test_run_testlib_runner_with_thread(self):
        print('test on running TESTLIB runner with thread...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = LibRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'thread': 3}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['TESTLIB_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['TESTLIB_score'], 99.946416604585025, places=4)

    def test_run_testlib_runner_phone_model(self):
        print('test on running TESTLIB runner with phone model...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = LibRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'phone_model': True}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['TESTLIB_score'], 92.54239166666667, places=3)
        self.assertAlmostEqual(results[1]['TESTLIB_score'], 100.0, places=4)
