from __future__ import absolute_import

import unittest

from vmaf.core.executor import run_executors_in_parallel
from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor, \
    NiqeNorefFeatureExtractor, BrisqueNorefFeatureExtractor

from .testutil import set_default_576_324_videos_for_testing

__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class NorefFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
            pass

    def test_noref_moment_fextractor(self):
        print('test on running Moment noref feature extractor on Assets...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

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
        print('test on running Moment noref feature extractor on NorefAssets...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

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

    def test_run_noref_brisque_fextractor(self):
        print('test on running BRISQUE noref feature extractor...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = BrisqueNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )

        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['BRISQUE_noref_feature_alpha23_score'], 0.78020833333333384, places=4)
        self.assertAlmostEqual(results[0]['BRISQUE_noref_feature_alpha13_score'], 0.6322500000000002, places=4)
        self.assertAlmostEqual(results[0]['BRISQUE_noref_feature_N34_score'],    -0.0071207420215536723, places=4)

        self.assertAlmostEqual(results[1]['BRISQUE_noref_feature_alpha23_score'], 0.87156250000000046, places=4)
        self.assertAlmostEqual(results[1]['BRISQUE_noref_feature_alpha13_score'], 0.82906250000000103, places=4)
        self.assertAlmostEqual(results[1]['BRISQUE_noref_feature_N34_score'],     -0.0092448158862212092, places=4)

    def test_run_noref_niqe_fextractor(self):
        print('test on running NIQE noref feature extractor...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = NiqeNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )

        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha23_score'], 0.8168807870370377, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha13_score'], 0.6949641203703707, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha_m1_score'], 2.0924143518518536, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_blbr1_score'], 0.72958634325785898, places=4)

        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha23_score'], 0.89566087962963026, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha13_score'], 0.85539583333333391, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha_m1_score'], 2.7192025462962985, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_blbr1_score'], 0.98723051960738684, places=4)

    def test_run_noref_niqe_fextractor_train(self):
        print('test on running NIQE noref feature extractor in train mode...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = NiqeNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'mode': 'train'},
            optional_dict2=None,
        )

        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha23_score'], 0.97259000000000073, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha13_score'], 0.80907000000000051, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha_m1_score'], 2.6135250000000019, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_blbr1_score'], 0.9150526409258144, places=4)

        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha23_score'], 0.97447727272727347, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha13_score'], 0.89120909090909162, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha_m1_score'], 3.0300909090909118, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_blbr1_score'], 1.0508255408831713, places=4)

    def test_run_noref_niqe_fextractor_with_patch_size(self):
        print('test on running NIQE noref feature extractor with custom patch size...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = NiqeNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'patch_size': 48},
            optional_dict2=None,
        )

        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha23_score'], 0.8430156250000006, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha13_score'], 0.71714583333333359, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha_m1_score'], 2.2195590277777795, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_blbr1_score'], 0.74061215376929412, places=4)

        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha23_score'], 0.9144918981481488, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha13_score'], 0.87132291666666728, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha_m1_score'], 2.8193532986111136, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_blbr1_score'], 0.99354006450609134, places=4)


class ParallelNorefFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractors'):
            for fextractor in self.fextractors:
                fextractor.remove_results()
            pass

    def test_run_parallel_moment_noref_fextractor(self):
        print('test on running Moment noref feature extractor on NorefAssets in parallel...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        fextractor = MomentNorefFeatureExtractor(
            [asset, asset_original],
            None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
        )
        self.fextractors = [fextractor]
        fextractor.run(parallelize=True)
        results = fextractor.results

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
        print('test on running Moment noref feature extractor on Assets...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

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
        print('test on running Moment noref feature extractor on NorefAssets...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

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

    def test_run_parallel_brisque_noref_fextractor(self):
        print('test on running BRISQUE noref feature extractor on NorefAssets in parallel...')
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractors, results = run_executors_in_parallel(
            BrisqueNorefFeatureExtractor,
            [asset, asset_original],
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=None,
        )

        self.assertAlmostEqual(results[0]['BRISQUE_noref_feature_alpha23_score'], 0.78020833333333384, places=4)
        self.assertAlmostEqual(results[0]['BRISQUE_noref_feature_alpha13_score'], 0.6322500000000002, places=4)
        self.assertAlmostEqual(results[0]['BRISQUE_noref_feature_N34_score'],     -0.0071207420215536723, places=4)

        self.assertAlmostEqual(results[1]['BRISQUE_noref_feature_alpha23_score'], 0.87156250000000046, places=4)
        self.assertAlmostEqual(results[1]['BRISQUE_noref_feature_alpha13_score'], 0.82906250000000103, places=4)
        self.assertAlmostEqual(results[1]['BRISQUE_noref_feature_N34_score'],     -0.0092448158862212092, places=4)

if __name__ == '__main__':
    unittest.main()
