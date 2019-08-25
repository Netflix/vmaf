import sys
import unittest

from vmaf.config import VmafConfig, VmafExternalConfig
from vmaf.core.asset import NorefAsset
from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


# @unittest.skipIf(not VmafExternalConfig.ffmpeg_path(), "ffmpeg not installed")
class NorefFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
            pass

    @unittest.skipIf(sys.version_info > (3,), reason="TODO python3: unclear how YuvReader is supposed to work")
    def test_noref_moment_fextractor_with_noref_asset_notyuv(self):
        print('test on running Moment noref feature extractor on NorefAssets (non-YUV)...')
        dis_path = VmafConfig.test_resource_path("mp4", "Seeking_10_288_375.mp4")
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      dis_path=dis_path,
                      asset_dict={'yuv_type': 'notyuv',
                                  'quality_width': 720, 'quality_height': 480,
                                  })

        self.fextractor = MomentNorefFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 63.776442013888882, places=4)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 5194.9118422453694, places=4)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1118.4952858425261, places=4)

    @unittest.skipIf(sys.version_info > (3,), reason="TODO python3: unclear how YuvReader is supposed to work")
    def test_noref_moment_fextractor_frames(self):
        print('test on running Moment noref feature extractor on Assets with frames...')
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'start_frame':2, 'end_frame':2,
                                  })

        self.fextractor = MomentNorefFeatureExtractor(
            [asset],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 62.315495327503427, places=4)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 4888.7623296039092, places=4)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1005.5413716918079, places=4)
