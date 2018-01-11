import unittest
from vmaf.config import VmafConfig, VmafExternalConfig
from vmaf.core.asset import Asset
from vmaf.core.feature_extractor import VmafFeatureExtractor, StrredFeatureExtractor
from vmaf.tools.stats import ListStats

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


@unittest.skipIf(not VmafExternalConfig.matlab_path(), "matlab not installed")
class ParallelMatlabFeatureExtractorTestNew(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass

    def test_run_strred_fextractor(self):
        print 'test on running STRRED feature extractor...'
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

        self.fextractor = StrredFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['STRRED_feature_srred_score'], 3.0114681041666671, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_trred_score'], 7.3039486249999994, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_strred_score'], 21.995608318659482, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_strred_score'], 0.0, places=4)

    def test_run_strred_fextractor_blackframes(self):
        print 'test on running STRRED feature extractor on flat frames...'
        ref_path = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "flat_1920_1080_10.yuv")
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

        from vmaf.core.result_store import FileSystemResultStore
        result_store = FileSystemResultStore(logger=None)

        self.fextractor = StrredFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=result_store
        )

        print '    running for the first time with fresh calculation...'
        self.fextractor.run(parallelize=True)

        result0, result1 = self.fextractor.results
        import os
        self.assertTrue(os.path.exists(result_store._get_result_file_path(result0)))
        self.assertTrue(os.path.exists(result_store._get_result_file_path(result1)))

        print '    running for the second time with stored results...'
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


class ParallelFeatureExtractorTestNew(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass

    def test_run_vmaf_fextractor_with_resampling(self):
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':160, 'quality_height':90})

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':160, 'quality_height':90})

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=False)

        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.74165043750000004, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'],1.4066421666666666, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.9808006041666667, places=4)
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
                      asset_dict={'width':576, 'height':324,
                                  'crop_cmd':'288:162:144:81',
                                  'quality_width':288, 'quality_height':162,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'crop_cmd':'288:162:144:81',
                                  'quality_width':288, 'quality_height':162,
                                  })

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=False)

        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.45365762500000012, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 2.8779373333333331, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.93894987889874548, places=4)
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
                      asset_dict={'width':576, 'height':324,
                                  'pad_cmd': 'iw+100:ih+100:50:50',
                                  'quality_width':676, 'quality_height':424,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'pad_cmd': 'iw+100:ih+100:50:50',
                                  'quality_width':676, 'quality_height':424,
                                  })

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=True)

        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.51023564583333325, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 2.6397702083333332, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.94112949409549829, places=4)
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
                      asset_dict={'width':576, 'height':324,
                                  'crop_cmd':'288:162:144:81',
                                  'pad_cmd': 'iw+288:ih+162:144:81',
                                  'quality_width':576, 'quality_height':324,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'crop_cmd':'288:162:144:81',
                                  'pad_cmd': 'iw+288:ih+162:144:81',
                                  'quality_width':576, 'quality_height':324,
                                  })

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=True)

        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.64106379166666672, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 0.7203213958333331, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.94700196017089999, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 32.78451041666667, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 0.7203213958333331, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 40.280504208333333, places=4)


if __name__ == '__main__':
    unittest.main()
