__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import sys
import unittest

from vmaf.core.train_test_model import SklearnRandomForestTrainTestModel, LibsvmNusvrTrainTestModel, \
    SklearnExtraTreesTrainTestModel
from vmaf.core.cross_validation import ModelCrossValidation
from vmaf.config import VmafConfig
from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor
from vmaf.routine import read_dataset
from vmaf.tools.misc import import_python_file, unroll_dict_of_lists


class CrossValidationTest(unittest.TestCase):

    def setUp(self):

        train_dataset_path = VmafConfig.test_resource_path('test_image_dataset_diffdim2.py')
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        runner = MomentNorefFeatureExtractor(
            train_assets,
            None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=None,
        )
        runner.run(parallelize=True)
        self.features = runner.results

    def test_run_cross_validation(self):

        train_test_model_class = SklearnRandomForestTrainTestModel
        model_param = {'norm_type': 'normalize',
                       'n_estimators': 10,
                       'random_state': 0}

        indices_train = range(9)
        indices_test = range(9)

        output = ModelCrossValidation.run_cross_validation(
            train_test_model_class, model_param, self.features,
            indices_train, indices_test)
        self.assertAlmostEqual(output['stats']['SRCC'], 0.93333333333333324, places=4)
        self.assertAlmostEqual(output['stats']['PCC'], 0.97754442316039469, places=4)
        self.assertAlmostEqual(output['stats']['KENDALL'], 0.83333333333333337, places=4)
        self.assertAlmostEqual(output['stats']['RMSE'], 0.17634739353518517, places=4)
        self.assertEqual(output['model'].TYPE, "RANDOMFOREST")

    def test_run_kfold_cross_validation_randomforest(self):

        train_test_model_class = SklearnRandomForestTrainTestModel
        model_param = {'norm_type': 'normalize',
                       'n_estimators': 10,
                       'random_state': 0}

        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, self.features, 3)

        self.assertAlmostEqual(output['aggr_stats']['SRCC'], 0.28452131897694583, places=4)
        self.assertAlmostEqual(output['aggr_stats']['PCC'], 0.1689046198483892, places=4)
        self.assertAlmostEqual(output['aggr_stats']['KENDALL'], 0.084515425472851652, places=4)
        self.assertAlmostEqual(output['aggr_stats']['RMSE'], 1.344683833136588, places=4)

    def test_run_kfold_cross_validation_extratrees(self):

        train_test_model_class = SklearnExtraTreesTrainTestModel
        model_param = {'norm_type': 'normalize', 'random_state': 0, 'n_estimators': 10}

        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, self.features, 3)

        self.assertAlmostEqual(output['aggr_stats']['SRCC'], 0.17320508075688773, places=4)
        self.assertAlmostEqual(output['aggr_stats']['PCC'], 0.33023719320146966, places=4)
        self.assertAlmostEqual(output['aggr_stats']['KENDALL'], 0.14907119849998599, places=4)
        self.assertAlmostEqual(output['aggr_stats']['RMSE'], 1.3279056191361394, places=4)

    def test_run_kfold_cross_validation_libsvmnusvr(self):

        train_test_model_class = LibsvmNusvrTrainTestModel
        model_param = {'norm_type': 'normalize'}

        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, self.features, 3)

        self.assertAlmostEqual(output['aggr_stats']['SRCC'], 0.31666666666666665, places=4)
        self.assertAlmostEqual(output['aggr_stats']['PCC'], 0.33103132578536021, places=4)
        self.assertAlmostEqual(output['aggr_stats']['KENDALL'], 0.27777777777777779, places=4)
        self.assertAlmostEqual(output['aggr_stats']['RMSE'], 1.2855099934718619, places=4)

    def test_run_kfold_cross_validation_with_list_input(self):

        train_test_model_class = SklearnRandomForestTrainTestModel
        model_param = {'norm_type': 'normalize',
                       'n_estimators': 10,
                       'random_state': 0}

        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, self.features,
            [[0, 3, 8], [2, 1, 5], [4, 6, 7]])

        self.assertAlmostEqual(output['aggr_stats']['SRCC'], 0.18333333333333335, places=4)
        self.assertAlmostEqual(output['aggr_stats']['PCC'], 0.35513638509959689, places=4)
        self.assertAlmostEqual(output['aggr_stats']['KENDALL'], 0.1111111111111111, places=3)
        self.assertAlmostEqual(output['aggr_stats']['RMSE'], 1.2740400878438387, places=3)

    def test_unroll_dict_of_lists(self):
        model_param_search_range = {'norm_type': ['normalize', 'clip_0to1'],
                                    'n_estimators': [10, 50], 'random_state': [0]}

        dicts = unroll_dict_of_lists(model_param_search_range)

        expected_dicts = [
         {'norm_type': 'normalize', 'n_estimators': 10, 'random_state': 0},
         {'norm_type': 'clip_0to1', 'n_estimators': 10, 'random_state': 0},
         {'norm_type': 'normalize', 'n_estimators': 50, 'random_state': 0},
         {'norm_type': 'clip_0to1', 'n_estimators': 50, 'random_state': 0},
        ]

        self.assertEqual(dicts, expected_dicts)

    @unittest.skipIf(sys.version_info < (3,), reason="For py3 only: py2 uses a different random seed.")
    def test_sample_model_param_list(self):
        import random
        random.seed(0)

        model_param_search_range = {'norm_type': ['normalize', 'clip_0to1'],
                                    'n_estimators': [10, 50], 'random_state': [0]}
        dicts = ModelCrossValidation._sample_model_param_list(model_param_search_range, 4)
        expected_dicts = [
            {'n_estimators': 50, 'norm_type': 'clip_0to1', 'random_state': 0},
            {'n_estimators': 50, 'norm_type': 'clip_0to1', 'random_state': 0},
            {'n_estimators': 50, 'norm_type': 'clip_0to1', 'random_state': 0},
            {'n_estimators': 10, 'norm_type': 'normalize', 'random_state': 0},
        ]
        for dict, expected_dict in zip(dicts, expected_dicts):
            self.assertDictEqual(dict, expected_dict)

        model_param_search_range = {'norm_type': ['normalize', 'clip_0to1'],
                                    'n_estimators': {'low': 10, 'high': 50, 'decimal': 0},
                                    'random_state': [0]}
        dicts = ModelCrossValidation._sample_model_param_list(
            model_param_search_range, 4)
        expected_dicts = [
            {'n_estimators': 15.0, 'norm_type': 'normalize', 'random_state': 0},
            {'n_estimators': 49.0, 'norm_type': 'normalize', 'random_state': 0},
            {'n_estimators': 13.0, 'norm_type': 'normalize', 'random_state': 0},
            {'n_estimators': 28.0, 'norm_type': 'normalize', 'random_state': 0},
        ]
        self.assertEqual(dicts, expected_dicts)

    def test_find_most_frequent_dict(self):
        dicts = [
         {'norm_type': 'normalize', 'n_estimators': 10, 'random_state': 0},
         {'norm_type': 'normalize', 'n_estimators': 50, 'random_state': 0},
         {'norm_type': 'clip_0to1', 'n_estimators': 10, 'random_state': 0},
         {'norm_type': 'clip_0to1', 'n_estimators': 50, 'random_state': 0},
         {'norm_type': 'clip_0to1', 'n_estimators': 50, 'random_state': 0},
        ]

        dict, count = ModelCrossValidation._find_most_frequent_dict(dicts)
        expected_dict = {'norm_type': 'clip_0to1', 'n_estimators': 50, 'random_state': 0}
        expected_count = 2

        self.assertEqual(dict, expected_dict)
        self.assertEqual(count, expected_count)

    def test_run_nested_kfold_cross_validation_randomforest(self):

        train_test_model_class = SklearnRandomForestTrainTestModel
        model_param_search_range = \
            {'norm_type': ['normalize'],
             'n_estimators': [10, 90],
             'max_depth': [None, 3],
             'random_state': [0]}

        output = ModelCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, self.features, 3)

        self.assertAlmostEqual(output['aggr_stats']['SRCC'], 0.40167715620274708, places=4)
        self.assertAlmostEqual(output['aggr_stats']['PCC'], 0.11009919053282299, places=4)
        self.assertAlmostEqual(output['aggr_stats']['KENDALL'], 0.14085904245475275, places=4)
        self.assertAlmostEqual(output['aggr_stats']['RMSE'], 1.3681348274719265, places=4)

        expected_top_model_param = {
            'norm_type': 'normalize',
            'n_estimators': 10,
            'max_depth': None,
            'random_state': 0
        }
        expected_top_ratio = 0.6666666666666666
        self.assertEqual(output['top_model_param'], expected_top_model_param)
        self.assertEqual(output['top_ratio'], expected_top_ratio)

    def test_run_nested_kfold_cross_validation_libsvmnusvr(self):

        train_test_model_class = LibsvmNusvrTrainTestModel
        model_param_search_range = \
            {'norm_type': ['normalize', 'clip_0to1', 'clip_minus1to1'],
             'kernel': ['rbf'],
             'nu': [0.5],
             'C': [1, 2],
             'gamma': [0.0]
             }

        output = ModelCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, self.features, 3)

        self.assertAlmostEqual(output['aggr_stats']['SRCC'], 0.30962614123961751, places=4)
        self.assertAlmostEqual(output['aggr_stats']['PCC'], -0.1535643705229309, places=4)
        self.assertAlmostEqual(output['aggr_stats']['KENDALL'], 0.14085904245475275, places=4)
        self.assertAlmostEqual(output['aggr_stats']['RMSE'], 1.5853397658781734, places=4)

        expected_top_model_param = {
            'norm_type': 'clip_0to1',
            'kernel': 'rbf',
            'nu': 0.5,
            'C': 1,
            'gamma': 0.0,
        }
        expected_top_ratio = 1.0

        self.assertEqual(output['top_model_param'], expected_top_model_param)
        self.assertEqual(output['top_ratio'], expected_top_ratio)

    def test_run_nested_kfold_cross_validation_with_list_input(self):

        train_test_model_class = SklearnRandomForestTrainTestModel
        model_param_search_range = \
            {'norm_type': ['none'],
             'n_estimators': [10, 90],
             'max_depth': [None, 3],
             'random_state': [0]
             }

        output = ModelCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, self.features,
            [[0, 3, 2], [8, 6, 5], [4, 1, 7]])

        self.assertAlmostEqual(output['aggr_stats']['SRCC'], 0.26666666666666666, places=4)
        self.assertAlmostEqual(output['aggr_stats']['PCC'], 0.15272340058922063, places=4)
        self.assertAlmostEqual(output['aggr_stats']['KENDALL'], 0.22222222222222221, places=4)
        self.assertAlmostEqual(output['aggr_stats']['RMSE'], 1.452887116343635, places=4)

        expected_top_model_param = {'norm_type': 'none',
                                    'n_estimators': 10,
                                    'max_depth': None,
                                    'random_state': 0
                                    }
        expected_top_ratio = 0.6666666666666666
        self.assertEqual(output['top_model_param'], expected_top_model_param)
        self.assertEqual(output['top_ratio'], expected_top_ratio)


if __name__ == '__main__':
    unittest.main(verbosity=2)
