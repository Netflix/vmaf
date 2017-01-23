import os
import unittest

import numpy as np

import config
from core.train_test_model import TrainTestModel, \
    LibsvmNusvrTrainTestModel, SklearnRandomForestTrainTestModel, \
    MomentRandomForestTrainTestModel, SklearnExtraTreesTrainTestModel
from core.executor import run_executors_in_parallel
from core.noref_feature_extractor import MomentNorefFeatureExtractor
from routine import read_dataset
from tools.misc import import_python_file
from core.raw_extractor import DisYUVRawVideoExtractor

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class TrainTestModelTest(unittest.TestCase):

    def setUp(self):

        train_dataset_path = config.ROOT + '/python/test/resource/test_image_dataset_diffdim.py'
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
        if hasattr(self, 'model'):
            self.model.delete(self.model_filename)

    def test_get_xs_ys(self):
        xs = TrainTestModel.get_xs_from_results(self.features, [0, 1, 2])

        self.assertEquals(len(xs['Moment_noref_feature_1st_score']), 3)
        self.assertAlmostEquals(np.mean(xs['Moment_noref_feature_1st_score']), 128.26146851380497, places=4)
        self.assertEquals(len(xs['Moment_noref_feature_var_score']), 3)
        self.assertAlmostEquals(np.mean(xs['Moment_noref_feature_var_score']), 1569.2395085695462, places=4)

        xs = TrainTestModel.get_xs_from_results(self.features)
        self.assertEquals(len(xs['Moment_noref_feature_1st_score']), 9)
        self.assertAlmostEquals(np.mean(xs['Moment_noref_feature_1st_score']), 111.59099599173773, places=4)
        self.assertEquals(len(xs['Moment_noref_feature_var_score']), 9)
        self.assertAlmostEquals(np.mean(xs['Moment_noref_feature_var_score']), 1806.8620377229011, places=4)

        ys = TrainTestModel.get_ys_from_results(self.features, [0, 1, 2])
        expected_ys = {'label': np.array([2.5, 3.9, 5.0]),
                       'content_id': np.array([0, 1, 2])}
        self.assertTrue(all(ys['label'] == expected_ys['label']))
        self.assertTrue(all(ys['content_id'] == expected_ys['content_id']))

    def test_train_save_load_predict(self):

        print "test train, save, load and predict..."

        xs = SklearnRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = SklearnRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = SklearnRandomForestTrainTestModel.get_xys_from_results(self.features)

        self.model = SklearnRandomForestTrainTestModel({'norm_type':'normalize', 'random_state':0}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)
        self.assertTrue(os.path.exists(self.model_filename))

        loaded_model = SklearnRandomForestTrainTestModel.from_file(self.model_filename, None)

        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.17634739353518517, places=4)

    def test_train_save_load_predict_libsvmnusvr(self):

        print "test libsvmnusvr train, save, load and predict..."

        xs = LibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = LibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = LibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = LibsvmNusvrTrainTestModel({'norm_type':'normalize'}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)
        self.assertTrue(os.path.exists(self.model_filename))
        self.assertTrue(os.path.exists(self.model_filename + '.model'))

        loaded_model = LibsvmNusvrTrainTestModel.from_file(self.model_filename, None)

        result = self.model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.62263086620058783, places=4)

        # loaded model generates slight numerical difference
        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.62263139871631323, places=4)

    def test_train_predict_libsvmnusvr(self):

        print "test libsvmnusvr train and predict..."

        # libsvmnusvr is bit exact to nusvr

        xs = LibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = LibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = LibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type':'normalize'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.62263086620058783, places=4)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type':'clip_0to1'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.84243141087114626, places=4)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'custom_clip_0to1_map': {
                'Moment_noref_feature_1st_score': [0.0, 100.0],
              },
             }, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.84639162766546994, places=4)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type':'clip_minus1to1'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.8314352752340991, places=4)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type':'none'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.23294283650716496, places=4)

    def test_train_predict_randomforest(self):

        print "test random forest train and predict..."

        # random forest don't need proper data normalization

        xs = SklearnRandomForestTrainTestModel.get_xs_from_results(self.features, [0, 1, 2])
        ys = SklearnRandomForestTrainTestModel.get_ys_from_results(self.features, [0, 1, 2])
        xys = SklearnRandomForestTrainTestModel.get_xys_from_results(self.features, [0, 1, 2])

        model = SklearnRandomForestTrainTestModel({'norm_type':'normalize',
                                            'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.051804171170643766, places=4)

        model = SklearnRandomForestTrainTestModel({'norm_type':'clip_0to1',
                                'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.051804171170643752, places=4)

        model = SklearnRandomForestTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'custom_clip_0to1_map': {
                'Moment_noref_feature_1st_score': [0.0, 100.0],
              },
             'random_state': 0
             }, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.051804171170643752, places=4)

        model = SklearnRandomForestTrainTestModel({'norm_type':'clip_minus1to1',
                                'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.051804171170643752, places=4)

        model = SklearnRandomForestTrainTestModel({'norm_type':'none', 'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.051804171170643752, places=4)

    def test_train_predict_extratrees(self):

        print "test extra trees train and predict..."

        # extra trees don't need proper data normalization

        xs = SklearnExtraTreesTrainTestModel.get_xs_from_results(self.features, [0, 1, 2])
        ys = SklearnExtraTreesTrainTestModel.get_ys_from_results(self.features, [0, 1, 2])
        xys = SklearnExtraTreesTrainTestModel.get_xys_from_results(self.features, [0, 1, 2])

        model = SklearnExtraTreesTrainTestModel({'norm_type':'normalize',
                                            'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.042867322777879642, places=4)

        model = SklearnExtraTreesTrainTestModel({'norm_type':'clip_0to1',
                                'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.042867322777879642, places=4)

        model = SklearnExtraTreesTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'custom_clip_0to1_map': {
                'Moment_noref_feature_1st_score': [0.0, 100.0],
              },
             'random_state': 0
             }, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.042867322777879642, places=4)

        model = SklearnExtraTreesTrainTestModel({'norm_type':'clip_minus1to1',
                                'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.042867322777879642, places=4)

        model = SklearnExtraTreesTrainTestModel({'norm_type':'none', 'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.042867322777879642, places=4)

class TrainTestModelWithDisYRawVideoExtractorTest(unittest.TestCase):

    def setUp(self):

        train_dataset_path = config.ROOT + '/python/test/resource/test_image_dataset_diffdim.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.h5py_filepath = config.ROOT + '/workspace/workdir/test.hdf5'
        self.h5py_file = DisYUVRawVideoExtractor.open_h5py_file(self.h5py_filepath)
        optional_dict2 = {'h5py_file': self.h5py_file}

        _, self.features = run_executors_in_parallel(
            DisYUVRawVideoExtractor,
            train_assets,
            fifo_mode=True,
            delete_workdir=True,
            parallelize=False, # CAN ONLY USE SERIAL MODE FOR DisYRawVideoExtractor
            result_store=None,
            optional_dict=None,
            optional_dict2=optional_dict2,
        )

        self.model_filename = config.ROOT + "/workspace/model/test_save_load.pkl"

    def tearDown(self):
        if hasattr(self, 'h5py_file'):
            DisYUVRawVideoExtractor.close_h5py_file(self.h5py_file)
        if os.path.exists(self.h5py_filepath):
            os.remove(self.h5py_filepath)
        if os.path.exists(self.model_filename):
            os.remove(self.model_filename)

    def test_extracted_features(self):
        self.assertAlmostEqual(np.mean(self.features[0]['dis_y']), 160.617204551784, places=4)
        self.assertAlmostEqual(np.mean(self.features[3]['dis_y']), 128.57907008374298, places=4)
        self.assertAlmostEqual(np.mean(self.features[0]['dis_u']), 135.38041204396345, places=4)
        self.assertAlmostEqual(np.mean(self.features[3]['dis_u']), 125.05257737968019, places=4)
        self.assertAlmostEqual(np.mean(self.features[0]['dis_v']), 125.26229752397977, places=4)
        self.assertAlmostEqual(np.mean(self.features[3]['dis_v']), 124.01385353721803, places=4)

    def test_get_xs_ys_consistent_wh(self):
        xs = MomentRandomForestTrainTestModel.get_xs_from_results(self.features, [0, 1, 2])
        ys = MomentRandomForestTrainTestModel.get_ys_from_results(self.features, [0, 1, 2])
        xys = MomentRandomForestTrainTestModel.get_xys_from_results(self.features, [0, 1, 2])

        expected_ys = {'label': [2.5, 3.9, 5.0], 'content_id': [0, 1, 2]}

        self.assertEqual(xs['dis_y'][0].shape, (1, 321, 481))
        self.assertEqual(xs['dis_y'][1].shape, (1, 321, 481))
        self.assertEqual(xs['dis_y'][2].shape, (1, 321, 481))
        self.assertEqual(xs['dis_u'][0].shape, (1, 321, 481))
        self.assertEqual(xs['dis_u'][1].shape, (1, 321, 481))
        self.assertEqual(xs['dis_u'][2].shape, (1, 321, 481))
        self.assertEqual(xs['dis_v'][0].shape, (1, 321, 481))
        self.assertEqual(xs['dis_v'][1].shape, (1, 321, 481))
        self.assertEqual(xs['dis_v'][2].shape, (1, 321, 481))
        self.assertAlmostEqual(np.mean(xs['dis_y']), 128.26146851380497, places=4)
        self.assertEqual(list(ys['label']), expected_ys['label'])
        self.assertEqual(list(ys['content_id']), expected_ys['content_id'])

        self.assertAlmostEqual(np.mean(xys['dis_y']), 128.26146851380497, places=4)
        self.assertEqual(list(xys['label']), expected_ys['label'])
        self.assertEqual(list(xys['content_id']), expected_ys['content_id'])

    def test_get_xs_ys_inconsistent_wh(self):
        xs = MomentRandomForestTrainTestModel.get_xs_from_results(self.features, [0, 8])
        ys = MomentRandomForestTrainTestModel.get_ys_from_results(self.features, [0, 8])
        xys = MomentRandomForestTrainTestModel.get_xys_from_results(self.features, [0, 8])

        expected_ys = {'label': [2.5, 4.7], 'content_id': [0, 13]}

        self.assertEqual(xs['dis_y'][0].shape, (1, 321, 481))
        self.assertEqual(xs['dis_y'][1].shape, (1, 486, 720))
        self.assertEqual(xs['dis_u'][0].shape, (1, 321, 481))
        self.assertEqual(xs['dis_u'][1].shape, (1, 486, 720))
        self.assertEqual(xs['dis_v'][0].shape, (1, 321, 481))
        self.assertEqual(xs['dis_v'][1].shape, (1, 486, 720))
        self.assertAlmostEqual(np.mean(xs['dis_y'][0]), 160.617204551784, places=4)
        self.assertAlmostEqual(np.mean(xs['dis_y'][1]), 35.8163894604481, places=4)
        self.assertEqual(list(ys['label']), expected_ys['label'])
        self.assertEqual(list(ys['content_id']), expected_ys['content_id'])

        self.assertEqual(xys['dis_y'][0].shape, (1, 321, 481))
        self.assertEqual(xys['dis_y'][1].shape, (1, 486, 720))
        self.assertEqual(xys['dis_u'][0].shape, (1, 321, 481))
        self.assertEqual(xys['dis_u'][1].shape, (1, 486, 720))
        self.assertEqual(xys['dis_v'][0].shape, (1, 321, 481))
        self.assertEqual(xys['dis_v'][1].shape, (1, 486, 720))
        self.assertEqual(list(xys['label']), expected_ys['label'])
        self.assertEqual(list(xys['content_id']), expected_ys['content_id'])

    def test_train_predict(self):
        xs = MomentRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = MomentRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = MomentRandomForestTrainTestModel.get_xys_from_results(self.features)

        # using dis_y only
        del xs['dis_u']
        del xs['dis_v']
        del xys['dis_u']
        del xys['dis_v']

        model = MomentRandomForestTrainTestModel({'norm_type':'normalize', 'random_state':0})
        model.train(xys)

        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.17634739353518517, places=4)

    def test_train_save_load_predict(self):

        xs = MomentRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = MomentRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = MomentRandomForestTrainTestModel.get_xys_from_results(self.features)

        # using dis_y only
        del xs['dis_u']
        del xs['dis_v']
        del xys['dis_u']
        del xys['dis_v']

        model = MomentRandomForestTrainTestModel({'norm_type':'normalize', 'random_state':0})
        model.train(xys)

        model.to_file(self.model_filename)
        self.assertTrue(os.path.exists(self.model_filename))

        loaded_model = TrainTestModel.from_file(self.model_filename)

        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.17634739353518517, places=4)

    def test_train_predict_using_yuv(self):
        xs = MomentRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = MomentRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = MomentRandomForestTrainTestModel.get_xys_from_results(self.features)

        model = MomentRandomForestTrainTestModel({'norm_type':'normalize', 'random_state':0})
        model.train(xys)

        result = model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.51128487038576109, places=4)


if __name__ == '__main__':
    unittest.main()
