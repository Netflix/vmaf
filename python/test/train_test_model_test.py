__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import unittest

import numpy as np

import config
from core.train_test_model import TrainTestModel, \
    LibsvmnusvrTrainTestModel, RandomForestTrainTestModel
from core.executor import run_executors_in_parallel
from core.noref_feature_extractor import MomentNorefFeatureExtractor
from routine import read_dataset
from tools.misc import import_python_file, empty_object

class TrainTestModelTest(unittest.TestCase):

    def setUp(self):

        train_dataset_path = config.ROOT + '/python/test/resource/test_image_dataset.py'
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

        self.model_filename = config.ROOT + "/workspace/model/test_save_load.pkl"

    def tearDown(self):
        if os.path.exists(self.model_filename):
            os.remove(self.model_filename)

    def test_get_xs_ys(self):
        xs = TrainTestModel.get_xs_from_results(self.features, [0, 1, 2])

        self.assertEquals(len(xs['Moment_noref_feature_1st_score']), 3)
        self.assertAlmostEquals(np.mean(xs['Moment_noref_feature_1st_score']), 128.26146851380497, places=4)
        self.assertEquals(len(xs['Moment_noref_feature_var_score']), 3)
        self.assertAlmostEquals(np.mean(xs['Moment_noref_feature_var_score']), 1569.2395085695462, places=4)

        xs = TrainTestModel.get_xs_from_results(self.features)
        self.assertEquals(len(xs['Moment_noref_feature_1st_score']), 8)
        self.assertAlmostEquals(np.mean(xs['Moment_noref_feature_1st_score']), 121.06282180814893, places=4)
        self.assertEquals(len(xs['Moment_noref_feature_var_score']), 8)
        self.assertAlmostEquals(np.mean(xs['Moment_noref_feature_var_score']), 1991.6255558247362, places=4)

        ys = TrainTestModel.get_ys_from_results(self.features, [0, 1, 2])
        expected_ys = {'label': np.array([2.5, 3.9, 5.0]),
                       'content_id': np.array([0, 1, 2])}
        self.assertTrue(all(ys['label'] == expected_ys['label']))
        self.assertTrue(all(ys['content_id'] == expected_ys['content_id']))

    def test_train_save_load_predict(self):

        print "test train, save, load and predict..."

        xs = TrainTestModel.get_xs_from_results(self.features)
        ys = TrainTestModel.get_ys_from_results(self.features)
        xys = TrainTestModel.get_xys_from_results(self.features)

        model = RandomForestTrainTestModel({'norm_type':'normalize', 'random_state':0}, None)
        model.train(xys)

        model.to_file(self.model_filename)
        self.assertTrue(os.path.exists(self.model_filename))

        loaded_model = RandomForestTrainTestModel.from_file(self.model_filename, None)

        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.15106640142026415, places=4)

        model.delete(self.model_filename)

    def test_train_save_load_predict_libsvmnusvr(self):

        print "test libsvmnusvr train, save, load and predict..."

        xs = TrainTestModel.get_xs_from_results(self.features)
        ys = TrainTestModel.get_ys_from_results(self.features)
        xys = TrainTestModel.get_xys_from_results(self.features)

        model = LibsvmnusvrTrainTestModel({'norm_type':'normalize'}, None)
        model.train(xys)

        model.to_file(self.model_filename)
        self.assertTrue(os.path.exists(self.model_filename))
        self.assertTrue(os.path.exists(self.model_filename + '.model'))

        loaded_model = LibsvmnusvrTrainTestModel.from_file(self.model_filename, None)

        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.39650837698192637, places=4)

        # loaded model generates slight numerical difference
        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.39650837698192637, places=4)

        model.delete(self.model_filename)

    def test_train_predict_libsvmnusvr(self):

        print "test libsvmnusvr train and predict..."

        # libsvmnusvr is bit exact to nusvr

        xs = TrainTestModel.get_xs_from_results(self.features)
        ys = TrainTestModel.get_ys_from_results(self.features)
        xys = TrainTestModel.get_xys_from_results(self.features)

        model = LibsvmnusvrTrainTestModel(
            {'norm_type':'normalize'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.39650837698192637, places=4)

        model = LibsvmnusvrTrainTestModel(
            {'norm_type':'clip_0to1'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.89839762651581445, places=4)

        model = LibsvmnusvrTrainTestModel(
            {'norm_type':'clip_minus1to1'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.81759338742714127, places=4)

        model = LibsvmnusvrTrainTestModel(
            {'norm_type':'none'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.21885719456686764, places=4)

    def test_train_predict_randomforest(self):

        print "test random forest train and predict..."

        # random forest don't need proper data normalization

        xs = TrainTestModel.get_xs_from_results(self.features, [0, 1, 2])
        ys = TrainTestModel.get_ys_from_results(self.features, [0, 1, 2])
        xys = TrainTestModel.get_xys_from_results(self.features, [0, 1, 2])

        model = RandomForestTrainTestModel({'norm_type':'normalize',
                                            'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.051804171170643766, places=4)

        model = RandomForestTrainTestModel({'norm_type':'clip_0to1',
                                'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.051804171170643752, places=4)

        model = RandomForestTrainTestModel({'norm_type':'clip_minus1to1',
                                'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.051804171170643752, places=4)

        model = RandomForestTrainTestModel({'norm_type':'none', 'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.051804171170643752, places=4)

if __name__ == '__main__':
    unittest.main()
