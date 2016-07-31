__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

from core.train_test_model import RandomForestTrainTestModel, LibsvmnusvrTrainTestModel
from core.cross_validation import ModelCrossValidation
import config
from core.executor import run_executors_in_parallel
from core.noref_feature_extractor import MomentNorefFeatureExtractor
from routine import read_dataset
from tools.misc import import_python_file

class FeatureCrossValidationTest(unittest.TestCase):

    def setUp(self):

        train_dataset_path = config.ROOT + '/python/test/resource/BSDS500_dataset.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        _, self.features = run_executors_in_parallel(
            MomentNorefFeatureExtractor,
            train_assets,
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=None,
        )

    def test_run_cross_validation(self):

        print "test cross validation..."

        train_test_model_class = RandomForestTrainTestModel
        model_param = {'norm_type':'normalize', 'random_state': 0}

        indices_train = range(5)
        indices_test = range(5)

        output = ModelCrossValidation.run_cross_validation(
            train_test_model_class, model_param, self.features,
            indices_train, indices_test)
        self.assertAlmostEquals(output['stats']['SRCC'], 1.0, places=4)
        self.assertAlmostEquals(output['stats']['PCC'], 0.98915736773708851, places=4)
        self.assertAlmostEquals(output['stats']['KENDALL'], 1.0, places=4)
        self.assertAlmostEquals(output['stats']['RMSE'], 0.26497837611988789, places=4)
        self.assertEquals(output['model'].TYPE, "RANDOMFOREST")

    def test_run_kfold_cross_validation_randomforest(self):

        print "test k-fold cross validation on random forest..."

        train_test_model_class = RandomForestTrainTestModel
        model_param = {'norm_type':'normalize', 'random_state': 0}

        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, self.features, 5)

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.8999999999999998, places=4)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.95636260884698576, places=4)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.79999999999999982, places=4)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.52346177899173219, places=4)

    def test_run_kfold_cross_validation_libsvmnusvr(self):

        print "test k-fold cross validation on libsvmnusvr..."

        train_test_model_class = LibsvmnusvrTrainTestModel
        model_param = {'norm_type': 'normalize'}

        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, self.features, 5)

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 1.0, places=4)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.95877329342682471, places=4)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 1.0, places=4)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.49959985942859997, places=4)

    def test_run_kfold_cross_validation_with_list_input(self):

        print "test k-fold cross validation with list input..."

        train_test_model_class = RandomForestTrainTestModel
        model_param = {'norm_type':'normalize', 'random_state': 0}

        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, self.features,
            [[0, 3], [2, 1], [4, ]])

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.69999999999999996, places=4)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.72607133059769025, places=4)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.59999999999999987, places=3)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 1.1817080521192824, places=3)

    def test_unroll_dict_of_lists(self):
        model_param_search_range = {'norm_type':['normalize', 'clip_0to1'],
                                    'n_estimators':[10, 50], 'random_state': [0]}

        dicts = ModelCrossValidation._unroll_dict_of_lists(model_param_search_range)

        expected_dicts = [
         {'norm_type':'normalize', 'n_estimators':10, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':10, 'random_state':0},
         {'norm_type':'normalize', 'n_estimators':50, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':50, 'random_state':0},
        ]

        self.assertEquals(dicts, expected_dicts)

    def test_sample_model_param_list(self):
        import random
        random.seed(0)

        model_param_search_range = {'norm_type':['normalize', 'clip_0to1'],
                                    'n_estimators':[10, 50], 'random_state': [0]}
        dicts = ModelCrossValidation._sample_model_param_list(
            model_param_search_range, 4)
        expected_dicts = [
         {'norm_type':'clip_0to1', 'n_estimators':50, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':10, 'random_state':0},
         {'norm_type':'normalize', 'n_estimators':50, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':50, 'random_state':0},
        ]
        self.assertEquals(dicts, expected_dicts)

        model_param_search_range = {'norm_type':['normalize', 'clip_0to1'],
                                    'n_estimators':{'low':10, 'high':50, 'decimal':0},
                                    'random_state': [0]}
        dicts = ModelCrossValidation._sample_model_param_list(
            model_param_search_range, 4)
        expected_dicts = [
         {'norm_type':'clip_0to1', 'n_estimators':21, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':20, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':42, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':39, 'random_state':0},
        ]
        self.assertEquals(dicts, expected_dicts)

    def test_find_most_frequent_dict(self):
        dicts = [
         {'norm_type':'normalize', 'n_estimators':10, 'random_state':0},
         {'norm_type':'normalize', 'n_estimators':50, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':10, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':50, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':50, 'random_state':0},
        ]

        dict, count = ModelCrossValidation._find_most_frequent_dict(dicts)
        expected_dict =  {'norm_type':'clip_0to1', 'n_estimators':50, 'random_state':0}
        expected_count = 2

        self.assertEquals(dict, expected_dict)
        self.assertEquals(count, expected_count)

    def test_run_nested_kfold_cross_validation_randomforest(self):

        print "test nested k-fold cross validation on random forest..."

        train_test_model_class = RandomForestTrainTestModel
        model_param_search_range = \
            {'norm_type':['normalize'],
             'n_estimators':[10, 90],
             'max_depth':[None, 3],
             'random_state': [0]}

        output = ModelCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, self.features, 5)

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.9, places=4)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.96225772597726533, places=4)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.79999999999999982, places=4)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.47861603467158437, places=4)

        expected_top_model_param = {'norm_type':'normalize',
                                'n_estimators':90,
                                'max_depth':None,
                                'random_state':0
                                }
        expected_top_ratio = 0.6
        self.assertEquals(output['top_model_param'], expected_top_model_param)
        self.assertEquals(output['top_ratio'], expected_top_ratio)

    def test_run_nested_kfold_cross_validation_libsvmnusvr(self):

        print "test nested k-fold cross validation on libsvmnusvr..."

        train_test_model_class = LibsvmnusvrTrainTestModel
        model_param_search_range = \
            {'norm_type':['normalize', 'clip_0to1', 'clip_minus1to1'],
             'kernel':['rbf'],
             'nu': [0.5, 1.0],
             'C': [1, 2],
             'gamma': [0.0]
             }

        output = ModelCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, self.features, 5)

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 1.0, places=4)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.96096817671162149, places=4)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 1.0, places=4)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.48460918155650223, places=4)

        expected_top_model_param = {'norm_type':'normalize',
                                'kernel':'rbf',
                                'nu':0.5,
                                'C':1,
                                'gamma':0.0,
                                }
        expected_top_ratio = 0.8

        self.assertEquals(output['top_model_param'], expected_top_model_param)
        self.assertEquals(output['top_ratio'], expected_top_ratio)

    def test_run_nested_kfold_cross_validation_with_list_input(self):

        print "test nested k-fold cross validation with list input..."

        train_test_model_class = RandomForestTrainTestModel
        model_param_search_range = \
            {'norm_type':['none'],
             'n_estimators':[10, 90],
             'max_depth':[None, 3],
             'random_state': [0]
             }

        output = ModelCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, self.features,
            [[0, 3], [2, 1], [4, ]]
        )

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.7, places=4)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.72607133059769058, places=4)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.6, places=4)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 1.1817080521192824, places=4)

        expected_top_model_param = {'norm_type':'none',
                                    'n_estimators':10,
                                    'max_depth':None,
                                    'random_state':0
                                    }
        expected_top_ratio = 1.0
        self.assertEquals(output['top_model_param'], expected_top_model_param)
        self.assertEquals(output['top_ratio'], expected_top_ratio)

if __name__ == '__main__':
    unittest.main()
