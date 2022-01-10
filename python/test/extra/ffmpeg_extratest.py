from __future__ import absolute_import

import unittest

from vmaf.config import VmafConfig, VmafExternalConfig
from vmaf.core.asset import Asset, NorefAsset
from vmaf.core.feature_extractor import VmafFeatureExtractor
from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor
from vmaf.core.quality_runner import VmafQualityRunner, PsnrQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from vmaf.tools.misc import MyTestCase


@unittest.skipIf(not VmafExternalConfig.ffmpeg_path() or 'apps' in VmafExternalConfig.ffmpeg_path(), 'ffmpeg not installed or ffmpeg should not be in apps')
class ParallelFeatureExtractorTestNew(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_run_vmaf_fextractor_with_gaussian_blurring(self):

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'crop_cmd': '288:162:144:81',
                                  'dis_gblur_cmd': 'sigma=0.01:steps=1',
                                  'quality_width': 288, 'quality_height': 162,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'crop_cmd': '288:162:144:81',
                                  'dis_gblur_cmd': 'sigma=0.01:steps=2',
                                  'quality_width': 288, 'quality_height': 162,
                                  })

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=True)

        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.45136466666666664, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 2.8779373333333331, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.936222875508755, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 24.109545916666665, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 0.9789283541666666, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 2.8779373333333331, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 0.9958889086049826, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.128021979166665, places=4)


@unittest.skipIf(not VmafExternalConfig.ffmpeg_path() or 'apps' in VmafExternalConfig.ffmpeg_path(), 'ffmpeg not installed or ffmpeg should not be in apps')
class NorefFeatureExtractorTest(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_noref_moment_fextractor_with_noref_asset_notyuv_gaussianblur(self):

        dis_path = VmafConfig.test_resource_path("mp4", "Seeking_10_288_375.mp4")
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      dis_path=dis_path,
                      asset_dict={'yuv_type': 'notyuv',
                                  'quality_width': 720, 'quality_height': 480,
                                  'gblur_cmd': 'sigma=0.01:steps=1',
                                  })

        self.fextractor = MomentNorefFeatureExtractor(
            [asset],
            None, fifo_mode=False,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 63.273976755401236, places=4)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 5124.572131500771, places=4)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1111.996326793719, places=4)


@unittest.skipIf(not VmafExternalConfig.ffmpeg_path() or 'apps' in VmafExternalConfig.ffmpeg_path(), 'ffmpeg not installed or ffmpeg should not be in apps')
class QualityRunnerTest(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
        super().tearDown()

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def test_run_psnr_runner_with_notyuv_gblur(self):

        ref_path = VmafConfig.test_resource_path("mp4", "Seeking_10_288_375.mp4")
        dis_path = VmafConfig.test_resource_path("mp4", "Seeking_10_288_375.mp4")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'yuv_type': 'notyuv',
                                  'quality_width': 720, 'quality_height': 480,
                                  'dis_gblur_cmd': 'sigma=0.01:steps=1',
                                  })
        self.runner = PsnrQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['PSNR_score'], 51.12088967333333, places=4)

    def test_run_vmaf_runner_with_notyuv_gblur(self):

        ref_path = VmafConfig.test_resource_path("mp4", "Seeking_30_480_1050.mp4")
        dis_path = VmafConfig.test_resource_path("mp4", "Seeking_10_288_375.mp4")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'yuv_type': 'notyuv',
                                  'quality_width': 360, 'quality_height': 240,
                                  'dis_gblur_cmd': 'sigma=0.01:steps=1',
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
        self.assertAlmostEqual(results[0]['VMAF_score'], 77.293043887001, places=0)

    def test_run_vmaf_runner_with_yuv_lutyuv(self):
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'quality_width': 360, 'quality_height': 240,
                                  'lutyuv_cmd': 'y=2*val',
                                  })
        self.runner = VmafQualityRunner(
            [asset],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1.json"),
            },
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VMAF_score'], 78.04870605403342, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
