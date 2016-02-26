__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
from train_test_model import RandomForestTrainTestModel, LibsvmnusvrTrainTestModel
from cross_validation import FeatureCrossValidation
import pandas as pd
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

        output = FeatureCrossValidation.run_cross_validation(
            train_test_model_class, model_param, feature_df,
            indices_train, indices_test)
        self.assertAlmostEquals(output['result']['SRCC'], 0.93180728084703823)
        self.assertAlmostEquals(output['result']['PCC'], 0.93897554632587299)
        self.assertAlmostEquals(output['result']['KENDALL'], 0.7809321265529332)
        self.assertAlmostEquals(output['result']['RMSE'], 0.32298193963956146)
        self.assertEquals(output['train_test_model'].TYPE, "RANDOMFOREST")

    def test_run_kfold_cross_validation_randomforest(self):

        print "test k-fold cross validation on random forest..."

        train_test_model_class = RandomForestTrainTestModel
        model_param = {'norm_type':'normalize', 'random_state': 0}

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        output = FeatureCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, feature_df, 6)

        self.assertAlmostEquals(output['aggregated_result']['SRCC'], 0.92541561799833183)
        self.assertAlmostEquals(output['aggregated_result']['PCC'], 0.93454811253831693)
        self.assertAlmostEquals(output['aggregated_result']['KENDALL'], 0.75702414684936781)
        self.assertAlmostEquals(output['aggregated_result']['RMSE'], 0.36158176166722)

    def test_run_kfold_cross_validation_libsvmnusvr(self):

        print "test k-fold cross validation on libsvmnusvr..."

        train_test_model_class = LibsvmnusvrTrainTestModel
        model_param = {'norm_type': 'normalize'}

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        output = FeatureCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, feature_df, 6)

        self.assertAlmostEquals(output['aggregated_result']['SRCC'], 0.92387451180595015)
        self.assertAlmostEquals(output['aggregated_result']['PCC'], 0.93031460919267095)
        self.assertAlmostEquals(output['aggregated_result']['KENDALL'], 0.75416215405673581)
        self.assertAlmostEquals(output['aggregated_result']['RMSE'], 0.3695746744572066)

    def test_run_kfold_cross_validation_with_list_input(self):

        print "test k-fold cross validation with list input..."

        train_test_model_class = RandomForestTrainTestModel
        model_param = {'norm_type':'normalize', 'random_state': 0}

        feature_df_file = config.ROOT + \
            "/python/test/resource/sample_feature_extraction_results.json"
        feature_df = pd.DataFrame.from_dict(eval(open(feature_df_file, "r").read()))

        feature_df = feature_df[:200]
        output = FeatureCrossValidation.run_kfold_cross_validation(
            train_test_model_class, model_param, feature_df,
            [range(0,50), range(130, 200), range(50, 130)])

        self.assertAlmostEquals(output['aggregated_result']['SRCC'], 0.91024373536714864)
        self.assertAlmostEquals(output['aggregated_result']['PCC'], 0.91972795735430202)
        self.assertAlmostEquals(output['aggregated_result']['KENDALL'], 0.73613552120569792)
        self.assertAlmostEquals(output['aggregated_result']['RMSE'], 0.42226137245391809)

    def test_unroll_dict_of_lists(self):
        model_param_search_range = {'norm_type':['normalize', 'clip_0to1'],
                                    'n_estimators':[10, 50], 'random_state': [0]}

        dicts = FeatureCrossValidation._unroll_dict_of_lists(model_param_search_range)

        expected_dicts = [
         {'norm_type':'normalize', 'n_estimators':10, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':10, 'random_state':0},
         {'norm_type':'normalize', 'n_estimators':50, 'random_state':0},
         {'norm_type':'clip_0to1', 'n_estimators':50, 'random_state':0},
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

        dict, count = FeatureCrossValidation._find_most_frequent_dict(dicts)
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

        output = FeatureCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, feature_df, 6)

        self.assertAlmostEquals(output['aggregated_result']['SRCC'], 0.93198668038126109)
        self.assertAlmostEquals(output['aggregated_result']['PCC'], 0.94632827102849348)
        self.assertAlmostEquals(output['aggregated_result']['KENDALL'], 0.77075614608695509)
        self.assertAlmostEquals(output['aggregated_result']['RMSE'], 0.32497387332652478)

        expected_model_param = {'norm_type':'normalize',
                                'n_estimators':90,
                                'max_depth':3,
                                'random_state':0
                                }
        expected_dominance = 0.6666666666666666
        self.assertEquals(output['dominated_model_param'], expected_model_param)
        self.assertEquals(output['model_param_dominance'], expected_dominance)

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

        output = FeatureCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, feature_df, 6)

        self.assertAlmostEquals(output['aggregated_result']['SRCC'], 0.93704238362264514)
        self.assertAlmostEquals(output['aggregated_result']['PCC'], 0.94445422982552052)
        self.assertAlmostEquals(output['aggregated_result']['KENDALL'], 0.77785381654919195)
        self.assertAlmostEquals(output['aggregated_result']['RMSE'], 0.33028967923638342)

        expected_model_param = {'norm_type':'clip_0to1',
                                'kernel':'rbf',
                                'nu':1.0,
                                'C':1,
                                'gamma':0.0,
                                }
        expected_dominance = 0.5

        self.assertEquals(output['dominated_model_param'], expected_model_param)
        self.assertEquals(output['model_param_dominance'], expected_dominance)

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
        output = FeatureCrossValidation.run_nested_kfold_cross_validation(
            train_test_model_class, model_param_search_range, feature_df,
            [range(0,50), range(130, 200), range(50, 130)]
        )

        self.assertAlmostEquals(output['aggregated_result']['SRCC'], 0.92795725021246278)
        self.assertAlmostEquals(output['aggregated_result']['PCC'], 0.93579312926288372)
        self.assertAlmostEquals(output['aggregated_result']['KENDALL'], 0.76612289581523185)
        self.assertAlmostEquals(output['aggregated_result']['RMSE'], 0.37876561452829105)

        expected_model_param = {'norm_type':'normalize',
                                'n_estimators':90,
                                'max_depth':3,
                                'random_state':0
                                }
        expected_dominance = 1.0
        self.assertEquals(output['dominated_model_param'], expected_model_param)
        self.assertEquals(output['model_param_dominance'], expected_dominance)
