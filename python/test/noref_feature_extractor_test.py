from __future__ import absolute_import

import unittest
from functools import partial

import numpy as np

from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor, \
    NiqeNorefFeatureExtractor, BrisqueNorefFeatureExtractor, SiTiNorefFeatureExtractor

from test.testutil import set_default_576_324_videos_for_testing, set_default_576_324_noref_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

from vmaf.core.result_store import FileSystemResultStore

from vmaf.tools.misc import MyTestCase


class NorefFeatureExtractorTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
            pass

    def test_noref_moment_fextractor(self):

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

        ref_path, dis_path, asset, asset_original = set_default_576_324_noref_videos_for_testing()

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

    def test_run_noref_brisque_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = BrisqueNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )

        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['BRISQUE_noref_feature_alpha23_score'], 0.7640625000000005, places=4)
        self.assertAlmostEqual(results[0]['BRISQUE_noref_feature_alpha13_score'], 0.6322500000000002, places=4)
        self.assertAlmostEqual(results[0]['BRISQUE_noref_feature_N34_score'],    -0.007239876204980851, places=4)

        self.assertAlmostEqual(results[1]['BRISQUE_noref_feature_alpha23_score'], 0.8644583333333339, places=4)
        self.assertAlmostEqual(results[1]['BRISQUE_noref_feature_alpha13_score'], 0.82906250000000103, places=4)
        self.assertAlmostEqual(results[1]['BRISQUE_noref_feature_N34_score'],     -0.0092448158862212092, places=4)

    def test_run_noref_niqe_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = NiqeNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None
        )

        self.fextractor.run(parallelize=True)

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

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = NiqeNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'mode': 'train'},
            optional_dict2=None,
        )

        self.fextractor.run(parallelize=True)

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

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = NiqeNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={'patch_size': 48},
            optional_dict2=None,
        )

        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha23_score'], 0.8430156250000006, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha13_score'], 0.71714583333333359, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_alpha_m1_score'], 2.2195590277777795, places=4)
        self.assertAlmostEqual(results[0]['NIQE_noref_feature_blbr1_score'], 0.74061215376929412, places=4)

        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha23_score'], 0.9144918981481488, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha13_score'], 0.87132291666666728, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_alpha_m1_score'], 2.8193532986111136, places=4)
        self.assertAlmostEqual(results[1]['NIQE_noref_feature_blbr1_score'], 0.99354006450609134, places=4)

    def test_noref_siti_fextractor_with_noref_asset(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_noref_videos_for_testing()

        self.fextractor = SiTiNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=True,
            result_store=None
        )
        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['SITI_noref_feature_si_score'], 79.27042052471928)
        self.assertAlmostEqual(results[0]['SITI_noref_feature_ti_score'], 13.580963712032636)

        self.assertAlmostEqual(results[1]['SITI_noref_feature_si_score'], 83.46568284569439)
        self.assertAlmostEqual(results[1]['SITI_noref_feature_ti_score'], 15.570422677885475)

        perc75 = partial(np.percentile, q=75)
        [result.set_score_aggregate_method(perc75) for result in results]

        self.assertAlmostEqual(results[0]['SITI_noref_feature_si_score'], 80.01507800178135)
        self.assertAlmostEqual(results[0]['SITI_noref_feature_ti_score'], 14.71399321502522)

        self.assertAlmostEqual(results[1]['SITI_noref_feature_si_score'], 84.00886573293045)
        self.assertAlmostEqual(results[1]['SITI_noref_feature_ti_score'], 16.862281850395433)

        perc100 = partial(np.percentile, q=100)
        [result.set_score_aggregate_method(perc100) for result in results]

        self.assertAlmostEqual(results[0]['SITI_noref_feature_si_score'], 82.84005099011819)
        self.assertAlmostEqual(results[0]['SITI_noref_feature_ti_score'], 17.60733889303054)

        self.assertAlmostEqual(results[1]['SITI_noref_feature_si_score'], 85.28055474796807)
        self.assertAlmostEqual(results[1]['SITI_noref_feature_ti_score'], 19.500837587311423)

    def test_noref_moment_fextractor_proc(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        callback_dict = {
            'dis_proc_callback': 'identity',
        }
        asset.asset_dict.update(callback_dict)
        asset_original.asset_dict.update(callback_dict)

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


class FeatureExtractorSaveWorkfilesTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'fextractor'):
            self.fextractor.remove_results()
        super().tearDown()

    def test_noref_moment_fextractor(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = MomentNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=self.result_store,
            save_workfiles=True,
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 61.332006624999984)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 4798.659574041666)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1036.8371843488285)

        self.assertAlmostEqual(results[1]['Moment_noref_feature_1st_score'], 59.788567297525134)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_2nd_score'], 4696.668388042271)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_var_score'], 1121.519917231207)

        self.fextractor.run()

    def test_noref_moment_fextractor_save_workfiles_second_time(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fextractor = MomentNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=self.result_store,
            save_workfiles=False,
        )
        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(results[0]['Moment_noref_feature_1st_score'], 61.332006624999984)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_2nd_score'], 4798.659574041666)
        self.assertAlmostEqual(results[0]['Moment_noref_feature_var_score'], 1036.8371843488285)

        self.assertAlmostEqual(results[1]['Moment_noref_feature_1st_score'], 59.788567297525134)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_2nd_score'], 4696.668388042271)
        self.assertAlmostEqual(results[1]['Moment_noref_feature_var_score'], 1121.519917231207)

        self.fextractor = MomentNorefFeatureExtractor(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=self.result_store,
            save_workfiles=True,
        )
        self.fextractor.run()


if __name__ == '__main__':
    unittest.main(verbosity=2)
