from vmaf.tools.stats import ListStats

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import unittest

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafLegacyQualityRunner, VmafQualityRunner, \
    PsnrQualityRunner, VmafossExecQualityRunner, MsSsimQualityRunner, \
    SsimQualityRunner, Adm2QualityRunner, VmafPhoneQualityRunner
from vmaf.core.result_store import FileSystemResultStore


class QualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_executor_id(self):
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480})
        runner = VmafLegacyQualityRunner([asset], None)
        self.assertEquals(runner.executor_id, 'VMAF_legacy_VF0.2.4b-1.1')

    def test_run_vamf_legacy_runner(self):
        print 'test on running VMAF (legacy) runner...'
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

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'],0.4460930625000001, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833337, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 65.393758021816708, places=4)
        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 96.444658329804156, places=4)

    def test_run_vamf_legacy_runner_10le(self):
        print 'test on running VMAF (legacy) runner on 10 bit le...'
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

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833333, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 65.393758021816708, places=4)
        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 96.444658329804156, places=4)

    def test_run_vamf_legacy_runner_with_result_store(self):
        print 'test on running VMAF (legacy) runner with result store...'
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

        result_store = FileSystemResultStore(logger=None)

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=result_store
        )

        print '    running for the first time with fresh calculation...'
        self.runner.run()
        result0, result1 = self.runner.results

        # NOTE: since stored results are actually VMAF_feature's not VMAF's,
        # the two paths below shouldn't exist
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertFalse(os.path.exists(result_store._get_result_file_path(result1)))

        print '    running for the second time with stored results...'
        self.runner.run()
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833333, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 65.393758021816708, places=4)
        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 96.444658329804156, places=4)

    def test_run_vmaf_runner_v1_model(self):
        print 'test on running VMAF runner (v1 model)...'
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

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath':VmafConfig.model_path("nflx_v1.pkl"),
            },
            optional_dict2=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 77.198367166977064, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_vmaf_runner(self):
        print 'test on running VMAF runner...'
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

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.93458780728708746, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.699271272486044, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_vmaf_runner_with_phone_score(self):
        print 'test on running VMAF runner with phone score...'
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

        self.runner = VmafQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAF_score'], 92.542390144364546, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_vmaf_phone_runner(self):
        print 'test on running VMAF phone runner...'
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

        with self.assertRaises(AssertionError):
            VmafPhoneQualityRunner(
                [asset, asset_original],
                None, fifo_mode=True,
                delete_workdir=True,
                result_store=None,
                optional_dict={
                    'enable_transform_score': True,
                }
            )

        self.runner = VmafPhoneQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_Phone_score'], 92.542390144364546, places=4)
        self.assertAlmostEqual(results[1]['VMAF_Phone_score'], 100.0, places=4)

    def test_run_vmaf_runner_checkerboard(self):
        print 'test on running VMAF runner on checkerboard pattern...'
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

        self.runner = VmafQualityRunner(
            [asset, asset_original, asset2],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.053996580527295335, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 0.999998395234, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.00000122625, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999998263056, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0000000801, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[2]['VMAF_feature_vif_scale0_score'], 0.112931470868, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_vif_scale1_score'], 0.298448603112, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_vif_scale2_score'], 0.337612207676, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_vif_scale3_score'], 0.496419716304, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[2]['VMAF_feature_adm2_score'], 0.7853384465157921, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 7.985956215118768, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.142659046424384, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 35.066157497128764, places=4)

    def test_run_vmaf_runner_flat(self):
        print 'test on running VMAF runner on flat pattern...'
        ref_path = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "flat_1920_1080_10.yuv")
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

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 97.427927701008869, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 97.428042675471147, places=4)

    def test_run_vmaf_runner_with_rf_model(self):
        print 'test on running VMAF runner with custom input model...'
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

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={
                'model_filepath':VmafConfig.model_path("nflx_vmaff_rf_v2.pkl"),
            },
            optional_dict2=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'],4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 79.224759615384599, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_vamf_runner_with_norm_type_none(self):
        print 'test on running VMAF quality runner with custom model...'
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

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={
                'model_filepath':VmafConfig.model_path("nflxtrain_norm_type_none.pkl"),
            },
            optional_dict2=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.93458780728708746, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 74.253349625150562, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 77.996338095161946, places=4)

    def test_run_psnr_runner(self):
        print 'test on running PSNR runner...'
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

        self.runner = PsnrQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)

        self.assertAlmostEqual(results[0]['PSNR_scores'][2], 30.993823, places=4)

    def test_run_vmafossexec_runner(self):
        print 'test on running VMAFOSSEXEC runner...'
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

    def test_run_vmafossexec_runner_with_phone_score(self):
        print 'test on running VMAFOSSEXEC runner with phone score...'
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

    def test_run_ssim_runner(self):
        print 'test on running SSIM runner...'
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

        self.runner = SsimQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['SSIM_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_l_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_c_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_s_score'], 0.89773633333333336, places=4)

        self.assertAlmostEqual(results[1]['SSIM_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_l_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_c_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_s_score'], 1.0, places=4)

    def test_run_ms_ssim_runner(self):
        print 'test on running MS-SSIM runner...'
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

        self.runner = MsSsimQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['MS_SSIM_score'], 0.9632498125, places=4)
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

        self.assertAlmostEqual(results[1]['MS_SSIM_score'], 1.0, places=4)
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

    def test_run_vmaf_runner_pool_harmonic_mean(self):
        print 'test on running VMAF runner (pool harmonic mean)...'
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

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        for result in results:
            result.set_score_aggregate_method(ListStats.harmonic_mean)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.36259426848527943, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.76614783252704499, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.86261788202730005, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.91587319935214206, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.5480006115798028, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.93454458698888132, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.5480006115798028, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.541148067714843, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.94504634354891, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 1.0, places=4)

    def test_run_vmaf_runner_pool_perc10(self):
        print 'test on running VMAF runner (pool 10-perctile)...'
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

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        for result in results:
            result.set_score_aggregate_method(ListStats.perc10)

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.3324451679468679, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.73826183605283979, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.84227039935569437, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.9023292478206113, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.3064741, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.92385012173707182, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.3064741, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)

        with self.assertRaises(KeyError):
            self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 72.753572476835515, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)

    def test_run_adm2_runner(self):
        print 'test on running ADM2 runner...'
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

        self.runner = Adm2QualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['ADM2_score'], 0.93458780728708746, places=4)
        self.assertAlmostEqual(results[1]['ADM2_score'], 1.0, places=4)

    def test_run_vmaf_runner_with_transform_score(self):
        print 'test on running VMAF runner with score transforming...'
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 32.757433750978919, places=4)

    def test_run_vmaf_runner_with_transform_score_2(self):
        print 'test on running VMAF runner with score transforming (2)...'
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafQualityRunner(
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

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], -7.2425662490210838, places=4)

    def test_run_vmaf_runner_with_transform_score_disabled(self):
        print 'test on running VMAF runner with score transforming disabled...'
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path("test_model_transform_add40.pkl"),
                'enable_transform_score': False,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale0_score'], 0.23738393128710478, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale1_score'], 0.08524788663335138, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale2_score'], 0.024058909404945077, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_scale3_score'], 0.018034879735107798, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 0.0, places=4)

    def test_run_vmaf_runner_with_transform_for_phone(self):
        print 'test on running VMAF runner with transform for phone...'
        ref_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_0_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "checkerboard_1920_1080_10_3_10_0.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=True,
            delete_workdir=True,
            optional_dict={
                'enable_transform_score': True,
            },
            result_store=self.result_store,
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.0, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 12.5548366667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.053996580527295335, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 14.982837886251568, places=4)

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

class ParallelQualityRunnerTest(unittest.TestCase):

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runners'):
            for runner in self.runners:
                runner.remove_results()
            pass

    def test_run_parallel_vmaf_legacy_runner(self):
        print 'test on running VMAF (legacy) quality runner in parallel...'
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

        self.runner = VmafLegacyQualityRunner(
            [asset, asset_original, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625000001, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833337, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

        self.assertAlmostEqual(results[0]['VMAF_legacy_score'], 65.393758021816708, places=4)
        self.assertAlmostEqual(results[1]['VMAF_legacy_score'], 96.444658329804156, places=4)
        self.assertAlmostEqual(results[2]['VMAF_legacy_score'], 96.444658329804156, places=4)

    def test_run_parallel_vmaf_runner(self):
        print 'test on running VMAF quality runner in parallel...'
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

        self.runner = VmafQualityRunner(
            [asset, asset_original, asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale0_score'], 0.363420489439, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale1_score'], 0.766647542135, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale2_score'], 0.862854666902, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_vif_scale3_score'], 0.915971778036, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.93458777083333333, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale0_score'], 1.00000001415, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale1_score'],0.99999972612, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale2_score'], 0.999999465724, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_vif_scale3_score'], 0.999999399683, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion2_score'], 3.8953518541666665, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.699271272486044, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.946416604585025, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 76.699271272486044, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 99.946416604585025, places=4)

    def test_run_parallel_psnr_runner(self):
        print 'test on running PSNR quality runner in parallel...'
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

        self.runner = PsnrQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)

    def test_run_parallel_vamf_runner_with_rf_model(self):
        print 'test on running VMAF quality runner in parallel with RF model...'
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

        self.runner = VmafQualityRunner(
            [asset, asset_original, asset, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath':VmafConfig.model_path("nflx_vmaff_rf_v2.pkl"),
            },
            optional_dict2=None,
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.4460930625, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.04982535417, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.93458777083333333, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.5095715208, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'],4.04982535417, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.2714392708, places=4)

        self.assertAlmostEqual(results[0]['VMAF_score'], 79.224759615384599, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 100.0, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 79.224759615384599, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 79.224759615384599, places=4)
        self.assertAlmostEqual(results[4]['VMAF_score'], 79.224759615384599, places=4)

    def test_run_parallel_ssim_runner(self):
        print 'test on running SSIM quality runner in parallel...'
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

        self.runner = SsimQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['SSIM_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_l_score'], 0.9981474583333334, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_c_score'], 0.96126793750000006, places=4)
        self.assertAlmostEqual(results[0]['SSIM_feature_ssim_s_score'], 0.89773633333333336, places=4)

        self.assertAlmostEqual(results[1]['SSIM_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_l_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_c_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['SSIM_feature_ssim_s_score'], 1.0, places=4)

        self.assertAlmostEqual(results[2]['SSIM_score'], 0.86322654166666657, places=4)
        self.assertAlmostEqual(results[3]['SSIM_score'], 0.86322654166666657, places=4)

    def test_run_parallel_msssim_runner(self):
        print 'test on running MS-SSIM quality runner in parallel...'
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

        self.runner = MsSsimQualityRunner(
            [asset, asset_original, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['MS_SSIM_score'], 0.9632498125, places=4)
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

        self.assertAlmostEqual(results[1]['MS_SSIM_score'], 1.0, places=4)
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

        self.assertAlmostEqual(results[2]['MS_SSIM_score'], 0.9632498125, places=4)

    def test_run_parallel_msssim_runner_with_result_store(self):
        print 'test on running MS-SSIM quality runner in parallel...'
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

        self.runner = MsSsimQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['MS_SSIM_score'], 0.9632498125, places=4)
        self.assertAlmostEqual(results[1]['MS_SSIM_score'], 1.0, places=4)
        self.assertAlmostEqual(results[2]['MS_SSIM_score'], 0.9632498125, places=4)
        self.assertAlmostEqual(results[3]['MS_SSIM_score'], 0.9632498125, places=4)

    def test_run_parallel_runner_with_repeated_assets(self):
        print 'test on running PSNR quality runner in parallel with repeated assets...'
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

        self.runner = PsnrQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store
        )
        self.runner.run(parallelize=True)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)
        self.assertAlmostEqual(results[2]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[3]['PSNR_score'], 30.755063979166664, places=4)

    def test_run_parallel_runner_with_parallel_disabled(self):
        print 'test on running PSNR quality runner in parallel with parallelization disabled...'
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

        self.runner = PsnrQualityRunner(
            [asset, asset_original, asset, asset],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=False)
        results = self.runner.results

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)
        self.assertAlmostEqual(results[2]['PSNR_score'], 30.755063979166664, places=4)
        self.assertAlmostEqual(results[3]['PSNR_score'], 30.755063979166664, places=4)

    def test_run_parallel_vmafossexec_runner_with_repeated_assets(self):
        print 'test on running VMAFOSSEXEC quality runner in parallel with repeated assets...'
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

class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmaf_quality_runner_version(self):
        self.assertEquals(VmafQualityRunner.VERSION, 'F0.2.4b-0.6.1')
        self.assertEquals(VmafQualityRunner.ALGO_VERSION, 2)

    def test_vmafossexec_quality_runner_version(self):
        self.assertEquals(VmafossExecQualityRunner.VERSION, 'F0.2.4b-0.6.1')
        self.assertEquals(VmafossExecQualityRunner.ALGO_VERSION, 2)

if __name__ == '__main__':
    unittest.main()
