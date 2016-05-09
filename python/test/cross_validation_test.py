__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

import pandas as pd

from core.train_test_model import RandomForestTrainTestModel, LibsvmnusvrTrainTestModel
from core.cross_validation import ModelCrossValidation
import config


class FeatureCrossValidationTest(unittest.TestCase):

    def test_run_cross_validation(self):

        print "test cross validation..."

        train_test_model_class = RandomForestTrainTestModel
        model_param = {'norm_type':'normalize', 'random_state': 0}

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        indices_train = range(250)
        indices_test = range(250, 300)

        output = ModelCrossValidation.run_cross_validation(
            train_test_model_class, model_param, feature_df,
            indices_train, indices_test)
        self.assertAlmostEquals(output['stats']['SRCC'], 0.93493301443051136)
        self.assertAlmostEquals(output['stats']['PCC'], 0.9413390374529329)
        self.assertAlmostEquals(output['stats']['KENDALL'], 0.78029280419726044)
        self.assertAlmostEquals(output['stats']['RMSE'], 0.32357946626958406)
        self.assertEquals(output['model'].TYPE, "RANDOMFOREST")

    def test_run_kfold_cross_validation_randomforest(self):

        print "test k-fold cross validation on random forest..."

        train_test_model_class = RandomForestTrainTestModel
        model_param = {'norm_type':'normalize', 'random_state': 0}

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, feature_df, 6)

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.92695443548602008)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.93189074441713937)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.76031309571294092)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.40381451586590256)

    def test_run_kfold_cross_validation_libsvmnusvr(self):

        print "test k-fold cross validation on libsvmnusvr..."

        train_test_model_class = LibsvmnusvrTrainTestModel
        model_param = {'norm_type': 'normalize'}

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, feature_df, 6)

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.92387451180595015)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.92481147926825724)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.75416215405673581)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.42231775639097513)

    def test_run_kfold_cross_validation_with_list_input(self):

        print "test k-fold cross validation with list input..."

        train_test_model_class = RandomForestTrainTestModel
        model_param = {'norm_type':'normalize', 'random_state': 0}

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        feature_df = feature_df[:200]
        output = ModelCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, feature_df,
            [range(0,50), range(130, 200), range(50, 130)])

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.90636761259756715)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.90819953685397914)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.72937284548325965)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.49899297305829415)

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
             'random_state': [0]
             }

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        output = ModelCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, feature_df, 6)

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.92805802153293737)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.94066838465382363)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.76196220071567478)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.37660623901376861)

        expected_top_model_param = {'norm_type':'normalize',
                                'n_estimators':90,
                                'max_depth':None,
                                'random_state':0
                                }
        expected_top_ratio = 0.5
        # self.assertEquals(output['top_model_param'], expected_top_model_param)
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

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        output = ModelCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, feature_df, 6)

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.93704238362264514)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.94734024567912978)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.77785381654919195)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.34039563991411448)

        expected_top_model_param = {'norm_type':'clip_0to1',
                                'kernel':'rbf',
                                'nu':1.0,
                                'C':1,
                                'gamma':0.0,
                                }
        expected_top_ratio = 0.5

        self.assertEquals(output['top_model_param'], expected_top_model_param)
        self.assertEquals(output['top_ratio'], expected_top_ratio)

    def test_run_nested_kfold_cross_validation_with_list_input(self):

        print "test nested k-fold cross validation with list input..."

        train_test_model_class = RandomForestTrainTestModel
        model_param_search_range = \
            {'norm_type':['normalize'],
             'n_estimators':[10, 90],
             'max_depth':[None, 3],
             'random_state': [0]
             }

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        feature_df = feature_df[:200]
        output = ModelCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, feature_df,
            [range(0,50), range(130, 200), range(50, 130)]
        )

        self.assertAlmostEquals(output['aggr_stats']['SRCC'], 0.92549459243170684)
        self.assertAlmostEquals(output['aggr_stats']['PCC'], 0.93070443071372855)
        self.assertAlmostEquals(output['aggr_stats']['KENDALL'], 0.76385104263763215)
        self.assertAlmostEquals(output['aggr_stats']['RMSE'], 0.43223946862572299)

        expected_top_model_param = {'norm_type':'normalize',
                                    'n_estimators':90,
                                    'max_depth':3,
                                    'random_state':0
                                    }
        expected_top_ratio = 0.6666666666666666
        self.assertEquals(output['top_model_param'], expected_top_model_param)
        self.assertEquals(output['top_ratio'], expected_top_ratio)

if __name__ == '__main__':
    unittest.main()
