import unittest
import config
from core.asset import Asset
from core.feature_extractor import VmafFeatureExtractor, StrredFeatureExtractor

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class ParallelFeatureExtractorTestNew(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        pass

    def test_run_vmaf_fextractor_with_resampling(self):
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':160, 'quality_height':90})

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':160, 'quality_height':90})

        self.fextractor = VmafFeatureExtractor(
            [asset, asset_original], None, fifo_mode=False)

        self.fextractor.run(parallelize=False)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.782546520833, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'],1.3216766875, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm_score'], 0.966705166667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 28.0085990417, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 1.3216766875, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.811350125, places=4)

    def test_run_vmaf_fextractor_with_cropping(self):
        # crop_cmd: 288:162:144:81 - crop to 288x162 with upper-left pixel
        # starting at coordinate (144, 81)

        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'crop_cmd':'288:162:144:81',
                                  'quality_width':288, 'quality_height':162,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=config.ROOT + "/workspace/workdir",
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
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.91519481795660607, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.942050354166668, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 2.8779373333333331, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.71648420833333, places=4)

    def test_run_vmaf_fextractor_with_padding(self):
        # pad_cmd: iw+100:ih+100:50:50 - pad to (iw+100)x(ih+100), where iw is
        # input width, ih is input height, and starting point is (-50, -50)

        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'pad_cmd': 'iw+100:ih+100:50:50',
                                  'quality_width':676, 'quality_height':424,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=config.ROOT + "/workspace/workdir",
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
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.94050402763695473, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 26.893242291666667, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 2.6397702083333332, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 34.306043416666668, places=4)

    def test_run_vmaf_fextractor_with_cropping_and_padding_to_original_wh(self):
        # crop_cmd: 288:162:144:81 - crop to the center 288x162 image
        # pad_cmd: iw+288:ih+162:144:81 - pad back to the original size

        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'crop_cmd':'288:162:144:81',
                                  'pad_cmd': 'iw+288:ih+162:144:81',
                                  'quality_width':576, 'quality_height':324,
                                  })

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=config.ROOT + "/workspace/workdir",
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
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'],0.93767663859084349, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 32.78451041666667, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 0.7203213958333331, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 40.280504208333333, places=4)

    def test_run_strred_fextractor(self):
        print 'test on running STRRED feature extractor...'
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
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

        self.assertAlmostEqual(results[0]['STRRED_feature_srred_score'], 4.8845008541666664, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_trred_score'], 8.9429378333333336, places=4)
        self.assertAlmostEqual(results[0]['STRRED_feature_strred_score'], 44.002554138184131, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_srred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_trred_score'], 0.0, places=4)
        self.assertAlmostEqual(results[1]['STRRED_feature_strred_score'], 0.0, places=4)

if __name__ == '__main__':
    unittest.main()
