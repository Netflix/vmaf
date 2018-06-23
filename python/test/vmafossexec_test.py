import unittest
from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafossExecQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from testutil import set_default_576_324_videos_for_testing

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class VmafossexecQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_vmafossexec_runner(self):
        print 'test on running VMAFOSSEXEC runner...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'],99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_thread(self):
        print 'test on running VMAFOSSEXEC runner with thread...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'thread': 3}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'],99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_subsample(self):
        print 'test on running VMAFOSSEXEC runner with subsample...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'subsample': 5}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.954390000000018, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.742800000000003, places=4)

    def test_run_vmafossexec_runner_with_phone_score(self):
        print 'test on running VMAFOSSEXEC runner with phone score...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'enable_transform_score': True,
            }
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 92.542390144364546, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 100.0, places=4)

    def test_run_vmafossexec_runner_norm_type_none(self):
        print 'test on running VMAFOSSEXEC runner with norm type none...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath':VmafConfig.model_path("nflxtrain_norm_type_none.pkl"),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 74.253349625150562, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 77.996338095161946, places=4)

    def test_run_vmafossexec_runner_yuv422p10le(self):
        print 'test on running VMAFOSSEXEC runner on 10 bit le...'
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv422p10le.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv422p10le.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'yuv_type':'yuv422p10le'})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'yuv_type':'yuv422p10le'})

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.780577083333331, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'], 0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'], 0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 72.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699271272486044, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416604585025, places=4)

    def test_run_vmafossexec_runner_with_transform_score(self):
        print 'test on running VMAFOSSEXEC runner with score transforming...'
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40.pkl"),
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 32.757433333333331, places=4)

    def test_run_vmafossexec_runner_with_transform_score_2(self):
        print 'test on running VMAFOSSEXEC runner with score transforming...'
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40_outltein.pkl"),
                'enable_transform_score': True,
                'disable_clip_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], -7.2425766666666673, places=4)

    def test_run_vmafossexec_runner_with_transform_score_disabled(self):
        print 'test on running VMAFOSSEXEC runner with score transforming disabled...'
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'model_filepath':VmafConfig.test_resource_path("test_model_transform_add40.pkl"),
                'enable_transform_score':False,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 0.0, places=4)

    def test_run_vmafossexec_runner_with_transform_for_phone(self):
        print 'test on running VMAFOSSEXEC runner with transform for phone...'
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'enable_transform_score':True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.053996566666666669, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 14.982840000000001, places=4)

    def test_run_vmafossexec_runner_with_phone_model(self):
        print 'test on running VMAFOSSEXEC runner with transform for phone...'
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafossExecQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            optional_dict={
                'phone_model':True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.053996566666666669, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 14.982840000000001, places=4)

    def test_run_vmafossexec_runner_disable_avx_precise(self):
        print 'test on running VMAFOSSEXEC runner disabling AVX (precise)...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'disable_avx': True}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953527083333337, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.96324068749999991, places=5)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.999999958333, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.999999416667, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953527083333337, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=5)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699306249999992, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416666666664, places=5)

    def test_run_vmafossexec_runner_enable_avx_precise(self):
        print 'test on running VMAFOSSEXEC runner enabling AVX (precise)...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.36342081250000002, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.76664741666666669, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.86285333333333336, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.91597195833333345, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953527083333337, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=5)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.96324068749999991, places=5)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.99999985416666659, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.99999960416666667, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.99999914583333338, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953527083333337, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=5)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699266666666674, places=5)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416666666664, places=5)

    def test_run_vmafossexec_runner_with_motion2(self):
        print 'test on running VMAFOSSEXEC runner with motion2 feature...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath':VmafConfig.test_resource_path("test_motion2.pkl")
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953522916666672, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion_score'], 4.04982583333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953522916666672, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 78.532525000000007, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 97.089554166666673, places=4)

    def test_run_vmafossexec_runner_with_ci(self):
        print 'test on running VMAFOSSEXEC runner with conf interval...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_rb_v0.6.2", "vmaf_rb_v0.6.2.pkl"),
                'ci': True
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale0_score'],0.363420458333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale1_score'], 0.766647520833, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale2_score'], 0.862854708333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_vif_scale3_score'], 0.915971791667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_adm2_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_psnr_score'], 30.7550666667, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ssim_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ms_ssim_score'], 0.9632498125, places=4)

        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale1_score'],0.999999958333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale2_score'],0.999999416667, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_vif_scale3_score'], 0.999999208333, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_psnr_score'], 60.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ssim_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ms_ssim_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 75.443043750000001, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.958047916666672, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_bagging_score'], 73.099946626689174, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_bagging_score'], 99.686116179979152, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_stddev_score'], 1.2301198477788975, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_stddev_score'], 1.5917514683608882, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ci95_low_score'], 70.801585803086553, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ci95_low_score'], 94.784491176494996, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_ci95_high_score'], 74.853442421187708, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_ci95_high_score'], 99.992560767034618, places=4)

    def test_run_vmafossexec_runner_with_ci_and_custom_model(self):
        print 'test on running VMAFOSSEXEC runner with conf interval and custom model...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2boot_test.pkl'),
                'ci': True
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 75.443043750000001, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.958047916666672, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_bagging_score'], 75.13012623785923, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_bagging_score'], 99.96504855577571, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_stddev_score'], 0.6812993325967104, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_stddev_score'], 0.03947607207290399, places=4)

    def test_run_vmafossexec_runner_with_ci_and_phone_model(self):
        print 'test on running VMAFOSSEXEC runner with conf interval and phone model...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_rb_v0.6.2", "vmaf_rb_v0.6.2.pkl"),
                'phone_model':True,
                'ci': True,
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 91.723012127641823, places=4)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_bagging_score'], 90.129761531349985, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_bagging_score'], 100.0, places=4)
        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_stddev_score'], 0.85880437658259945, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_stddev_score'], 0.0, places=4)

class VmafossexecQualityRunnerSubsamplingTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner0'):
            self.runner0.remove_results()
        if hasattr(self, 'runner'):
            self.runner.remove_results()

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_vmafossexec_runner_with_subsample2(self):
        print 'test on running VMAFOSSEXEC runner with subsample2...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        subsample = 5

        self.runner0 = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={}
        )
        self.runner0.run()
        results0 = self.runner0.results

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'subsample': subsample}
        )
        self.runner.run()
        results = self.runner.results

        for i in range(48):
            if i % subsample == 0:
                self.assertAlmostEqual(results0[0]['VMAFOSSEXEC_scores'][i], results[0]['VMAFOSSEXEC_scores'][i / subsample], places=7)
                self.assertAlmostEqual(results0[1]['VMAFOSSEXEC_scores'][i], results[1]['VMAFOSSEXEC_scores'][i / subsample], places=7)


class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmafossexec_quality_runner_version(self):
        self.assertEquals(VmafossExecQualityRunner.VERSION, 'F0.2.4c-0.6.1')
        self.assertEquals(VmafossExecQualityRunner.ALGO_VERSION, 2)


class ParallelQualityRunnerTest(unittest.TestCase):

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runners'):
            for runner in self.runners:
                runner.remove_results()
            pass

    def test_run_parallel_vmafossexec_runner_with_repeated_assets(self):
        print 'test on running VMAFOSSEXEC quality runner in parallel with repeated assets...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafossExecQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFOSSEXEC_score'], 76.699266666666674, places=3)
        self.assertAlmostEqual(results[1]['VMAFOSSEXEC_score'], 99.946416666666664, places=4)
        self.assertAlmostEqual(results[2]['VMAFOSSEXEC_score'], 76.699266666666674, places=3)
        self.assertAlmostEqual(results[3]['VMAFOSSEXEC_score'], 76.699266666666674, places=3)


if __name__ == '__main__':
    unittest.main()
