from core.executor import run_executors_in_parallel

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

import config
from core.asset import NorefAsset, Asset
from core.noref_feature_extractor import MomentNorefFeatureExtractor

class NorefFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
            pass

    def test_noref_moment_fextractor(self):
        print 'test on running Moment noref feature extractor on Assets...'
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

        self.fextractor = MomentNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 61.332006624999984)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 4798.659574041666)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1036.8371843488285)

        self.assertAlmostEqual(results[1]['Moment_noref_feature_1st_score'], 59.788567297525134)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_2nd_score'], 4696.668388042271)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_var_score'], 1121.519917231207)

    def test_noref_moment_fextractor_with_noref_asset(self):
        print 'test on running Moment noref feature extractor on NorefAssets...'
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = NorefAsset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        self.fextractor = MomentNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 61.332006624999984)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 4798.659574041666)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1036.8371843488285)

        self.assertAlmostEqual(results[1]['Moment_noref_feature_1st_score'], 59.788567297525134)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_2nd_score'], 4696.668388042271)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_var_score'], 1121.519917231207)

class ParallelNorefFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractors'):
            for fextractor in self.fextractors:
                fextractor.remove_results()
            pass

    def test_run_parallel_moment_noref_fextractor(self):
        print 'test on running Moment noref feature extractor on NorefAssets in parallel...'
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = NorefAsset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})
        self.fextractors, results = run_executors_in_parallel(
            MomentNorefFeatureExtractor,
            [asset, asset_original],
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=None,
        )

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 61.332006624999984)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 4798.659574041666)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1036.8371843488285)

        self.assertAlmostEqual(results[1]['Moment_noref_feature_1st_score'], 59.788567297525134)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_2nd_score'], 4696.668388042271)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_var_score'], 1121.519917231207)

class ParallelNorefFeatureExtractorTestNew(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
            pass

    def test_noref_moment_fextractor(self):
        print 'test on running Moment noref feature extractor on Assets...'
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

        self.fextractor = MomentNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 61.332006624999984)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 4798.659574041666)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1036.8371843488285)

        self.assertAlmostEqual(results[1]['Moment_noref_feature_1st_score'], 59.788567297525134)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_2nd_score'], 4696.668388042271)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_var_score'], 1121.519917231207)

    def test_noref_moment_fextractor_with_noref_asset(self):
        print 'test on running Moment noref feature extractor on NorefAssets...'
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = NorefAsset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        self.fextractor = MomentNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 61.332006624999984)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 4798.659574041666)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1036.8371843488285)

        self.assertAlmostEqual(results[1]['Moment_noref_feature_1st_score'], 59.788567297525134)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_2nd_score'], 4696.668388042271)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_var_score'], 1121.519917231207)

if __name__ == '__main__':
    unittest.main()
