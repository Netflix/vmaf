import unittest

from vmaf import config
from vmaf.core.asset import NorefAsset
from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class NorefFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
            pass

    def test_noref_moment_fextractor_with_noref_asset_notyuv(self):
        print 'test on running Moment noref feature extractor on NorefAssets ' \
              '(non-YUV)...'
        dis_path = config.ROOT + "/python/test/resource/mp4/Seeking_10_288_375.mp4"
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
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

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 63.763580381944436, places=4)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 5158.9407083526221, places=4)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1084.1635630076041, places=4)
