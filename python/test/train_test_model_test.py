import os
import unittest

import numpy as np

from vmaf.config import VmafConfig
from vmaf.core.train_test_model import TrainTestModel, \
    LibsvmNusvrTrainTestModel, SklearnRandomForestTrainTestModel, \
    MomentRandomForestTrainTestModel, SklearnExtraTreesTrainTestModel, \
    SklearnLinearRegressionTrainTestModel, Logistic5PLRegressionTrainTestModel
from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor
from vmaf.routine import read_dataset
from vmaf.tools.misc import import_python_file
from vmaf.core.raw_extractor import DisYUVRawVideoExtractor

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class TrainTestModelTest(unittest.TestCase):

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

        self.model_filename = VmafConfig.workspace_path("model", "test_save_load.pkl")

    def tearDown(self):
        if hasattr(self, 'model'):
            self.model.delete(self.model_filename)

    def test_get_xs_ys(self):
        xs = TrainTestModel.get_xs_from_results(self.features, [0, 1, 2])

        self.assertEqual(len(xs['Moment_noref_feature_1st_score']), 3)
        self.assertAlmostEqual(np.mean(xs['Moment_noref_feature_1st_score']), 128.26146851380497, places=4)
        self.assertEqual(len(xs['Moment_noref_feature_var_score']), 3)
        self.assertAlmostEqual(np.mean(xs['Moment_noref_feature_var_score']), 1569.2395085695462, places=4)

        xs = TrainTestModel.get_xs_from_results(self.features)
        self.assertEqual(len(xs['Moment_noref_feature_1st_score']), 9)
        self.assertAlmostEqual(np.mean(xs['Moment_noref_feature_1st_score']), 111.59099599173773, places=4)
        self.assertEqual(len(xs['Moment_noref_feature_var_score']), 9)
        self.assertAlmostEqual(np.mean(xs['Moment_noref_feature_var_score']), 1806.8620377229011, places=4)

        ys = TrainTestModel.get_ys_from_results(self.features, [0, 1, 2])
        expected_ys = {'label': np.array([2.5, 3.9, 5.0]),
                       'content_id': np.array([0, 1, 2])}
        self.assertTrue(all(ys['label'] == expected_ys['label']))
        self.assertTrue(all(ys['content_id'] == expected_ys['content_id']))

    def test_train_save_load_predict(self):

        xs = SklearnRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = SklearnRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = SklearnRandomForestTrainTestModel.get_xys_from_results(self.features)

        self.model = SklearnRandomForestTrainTestModel({'norm_type': 'normalize',
                                                        'n_estimators': 10,
                                                        'random_state': 0}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)
        self.assertTrue(os.path.exists(self.model_filename))

        loaded_model = SklearnRandomForestTrainTestModel.from_file(self.model_filename, None)

        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.17634739353518517, places=4)

    def test_train_save_load_predict_libsvmnusvr(self):

        xs = LibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = LibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = LibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = LibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)
        self.assertTrue(os.path.exists(self.model_filename))
        self.assertTrue(os.path.exists(self.model_filename + '.model'))

        loaded_model = LibsvmNusvrTrainTestModel.from_file(self.model_filename, None)

        result = self.model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.62263086620058783, places=4)

        # loaded model generates slight numerical difference
        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.62263139871631323, places=4)

    def test_train_predict_libsvmnusvr(self):

        # libsvmnusvr is bit exact to nusvr

        xs = LibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = LibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = LibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type': 'normalize'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.62263086620058783, places=4)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type': 'clip_0to1'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.84243141087114626, places=4)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'custom_clip_0to1_map': {
                'Moment_noref_feature_1st_score': [0.0, 100.0],
              },
             }, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.84639162766546994, places=4)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type': 'clip_minus1to1'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.8314352752340991, places=4)

        model = LibsvmNusvrTrainTestModel(
            {'norm_type': 'none'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.23294283650716496, places=4)

    def test_train_across_test_splits_ci_libsvmnusvr(self):

        xs = LibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = LibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = LibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = LibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)

        ys_label_pred = self.model.predict(xs)['ys_label_pred']
        ys_label = ys['label']

        n_splits_test_indices = 3

        stats = self.model.get_stats(ys_label, ys_label_pred,
                                     split_test_indices_for_perf_ci=True,
                                     n_splits_test_indices=n_splits_test_indices)

        # check that the performance metric distributions have been passed out
        assert 'SRCC_across_test_splits_distribution' in stats, 'SRCC across_test_splits distribution non-existing.'
        assert 'PCC_across_test_splits_distribution' in stats, 'PCC across_test_splits distribution non-existing.'
        assert 'RMSE_across_test_splits_distribution' in stats, 'RMSE across_test_splits distribution non-existing.'

        # check that the length of the perf metrc lists is equal to n_splits_test_indices
        assert len(stats['SRCC_across_test_splits_distribution']) == n_splits_test_indices, \
            'SRCC list is not equal to the number of splits specified.'
        assert len(stats['PCC_across_test_splits_distribution']) == n_splits_test_indices, \
            'PCC list is not equal to the number of splits specified.'
        assert len(stats['RMSE_across_test_splits_distribution']) == n_splits_test_indices, \
            'RMSE list is not equal to the number of splits specified.'

        self.assertAlmostEqual(stats['SRCC_across_test_splits_distribution'][0], 0.391304347826087, places=4)
        self.assertAlmostEqual(stats['SRCC_across_test_splits_distribution'][1], 0.8983050847457626, places=4)
        self.assertAlmostEqual(stats['SRCC_across_test_splits_distribution'][2], 0.9478260869565218, places=4)

        self.assertAlmostEqual(stats['PCC_across_test_splits_distribution'][0], 0.554154433495891, places=4)
        self.assertAlmostEqual(stats['PCC_across_test_splits_distribution'][1], 0.817203617522247, places=4)
        self.assertAlmostEqual(stats['PCC_across_test_splits_distribution'][2], 0.5338890441054945, places=4)

        self.assertAlmostEqual(stats['RMSE_across_test_splits_distribution'][0], 0.6943573719335835, places=4)
        self.assertAlmostEqual(stats['RMSE_across_test_splits_distribution'][1], 0.5658403884750773, places=4)
        self.assertAlmostEqual(stats['RMSE_across_test_splits_distribution'][2], 0.5997800884581888, places=4)

        stats_no_test_split = self.model.get_stats(ys_label, ys_label_pred,
                                                   split_test_indices_for_perf_ci=False)

        # check that the performance metric distributions are not in these stats dict
        assert 'SRCC_across_test_splits_distribution' not in stats_no_test_split, \
            'SRCC across_test_splits distribution should not exist.'
        assert 'PCC_across_test_splits_distribution' not in stats_no_test_split, \
            'PCC across_test_splits distribution should not exist.'
        assert 'RMSE_across_test_splits_distribution' not in stats_no_test_split, \
            'RMSE across_test_splits distribution should not exist.'

    def test_train_predict_randomforest(self):

        # random forest don't need proper data normalization

        xs = SklearnRandomForestTrainTestModel.get_xs_from_results(self.features, [0, 1, 2])
        ys = SklearnRandomForestTrainTestModel.get_ys_from_results(self.features, [0, 1, 2])
        xys = SklearnRandomForestTrainTestModel.get_xys_from_results(self.features, [0, 1, 2])

        model = SklearnRandomForestTrainTestModel({'norm_type': 'normalize',
                                                   'n_estimators': 10,
                                                   'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.051804171170643766, places=4)

        model = SklearnRandomForestTrainTestModel({'norm_type': 'clip_0to1',
                                                   'n_estimators': 10,
                                                   'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.051804171170643752, places=4)

        model = SklearnRandomForestTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'n_estimators': 10,
             'custom_clip_0to1_map': {
                'Moment_noref_feature_1st_score': [0.0, 100.0],
              },
             'random_state': 0
             }, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.051804171170643752, places=4)

        model = SklearnRandomForestTrainTestModel({'norm_type': 'clip_minus1to1',
                                                   'n_estimators': 10,
                                                   'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.051804171170643752, places=4)

        model = SklearnRandomForestTrainTestModel({'norm_type': 'none',
                                                   'n_estimators': 10,
                                                   'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.051804171170643752, places=4)

    def test_train_predict_linearregression(self):

        # linear regression doesn't need proper data normalization

        xs = SklearnLinearRegressionTrainTestModel.get_xs_from_results(self.features, [0, 1, 2, 3, 4, 5])
        ys = SklearnLinearRegressionTrainTestModel.get_ys_from_results(self.features, [0, 1, 2, 3, 4, 5])
        xys = SklearnLinearRegressionTrainTestModel.get_xys_from_results(self.features, [0, 1, 2, 3, 4, 5])

        model = SklearnLinearRegressionTrainTestModel({'norm_type': 'normalize'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)

        self.assertAlmostEqual(result['RMSE'], 0.49489849608079006, places=4)

    def test_train_predict_extratrees(self):

        # extra trees don't need proper data normalization

        xs = SklearnExtraTreesTrainTestModel.get_xs_from_results(self.features, [0, 1, 2])
        ys = SklearnExtraTreesTrainTestModel.get_ys_from_results(self.features, [0, 1, 2])
        xys = SklearnExtraTreesTrainTestModel.get_xys_from_results(self.features, [0, 1, 2])

        model = SklearnExtraTreesTrainTestModel({'norm_type': 'normalize',
                                                 'n_estimators': 10,
                                                 'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.042867322777879642, places=4)

        model = SklearnExtraTreesTrainTestModel({'norm_type': 'clip_0to1',
                                                 'n_estimators': 10,
                                                 'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.042867322777879642, places=4)

        model = SklearnExtraTreesTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'n_estimators': 10,
             'custom_clip_0to1_map': {
                 'Moment_noref_feature_1st_score': [0.0, 100.0],
             },
             'random_state': 0
             }, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.042867322777879642, places=4)

        model = SklearnExtraTreesTrainTestModel({'norm_type': 'clip_minus1to1',
                                                 'n_estimators': 10,
                                                 'random_state': 0}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.042867322777879642, places=4)

        model = SklearnExtraTreesTrainTestModel({'norm_type': 'none',
                                                 'n_estimators': 10,
                                                 'random_state': 0,}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.042867322777879642, places=4)

    def test_train_logistic_fit_5PL(self):    
        xs = Logistic5PLRegressionTrainTestModel.get_xs_from_results(self.features, [0, 1, 2, 3, 4, 5], features=['Moment_noref_feature_1st_score'])
        ys = Logistic5PLRegressionTrainTestModel.get_ys_from_results(self.features, [0, 1, 2, 3, 4, 5])

        xys = {}
        xys.update(xs)
        xys.update(ys)
        
        model = Logistic5PLRegressionTrainTestModel({'norm_type': 'clip_0to1'}, None)
        model.train(xys)
        result = model.evaluate(xs, ys)

        self.assertAlmostEqual(result['RMSE'], 0.3603374311919728, places=4)


class TrainTestModelWithDisYRawVideoExtractorTest(unittest.TestCase):

    def setUp(self):

        train_dataset_path = VmafConfig.test_resource_path('test_image_dataset_diffdim2.py')
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.h5py_filepath = VmafConfig.workdir_path('test.hdf5')
        self.h5py_file = DisYUVRawVideoExtractor.open_h5py_file(self.h5py_filepath)
        optional_dict2 = {'h5py_file': self.h5py_file}

        runner = DisYUVRawVideoExtractor(
            train_assets,
            None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=optional_dict2,
        )
        runner.run(parallelize=False)  # CAN ONLY USE SERIAL MODE FOR DisYRawVideoExtractor
        self.features = runner.results

        self.model_filename = VmafConfig.workspace_path("model", "test_save_load.pkl")

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

        model = MomentRandomForestTrainTestModel({'norm_type': 'normalize',
                                                  'n_estimators': 10,
                                                  'random_state': 0})
        model.train(xys)

        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.17634739353518517, places=4)

    def test_train_save_load_predict(self):

        xs = MomentRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = MomentRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = MomentRandomForestTrainTestModel.get_xys_from_results(self.features)

        # using dis_y only
        del xs['dis_u']
        del xs['dis_v']
        del xys['dis_u']
        del xys['dis_v']

        model = MomentRandomForestTrainTestModel({'norm_type': 'normalize',
                                                  'n_estimators': 10,
                                                  'random_state': 0})
        model.train(xys)

        model.to_file(self.model_filename)
        self.assertTrue(os.path.exists(self.model_filename))

        loaded_model = TrainTestModel.from_file(self.model_filename)

        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.17634739353518517, places=4)

    def test_train_predict_using_yuv(self):
        xs = MomentRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = MomentRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = MomentRandomForestTrainTestModel.get_xys_from_results(self.features)

        model = MomentRandomForestTrainTestModel({'norm_type': 'normalize',
                                                  'n_estimators': 10,
                                                  'random_state': 0})
        model.train(xys)

        result = model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.51128487038576109, places=4)


class TrainTestModelTestJson(unittest.TestCase):

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
        runner.run(parallelize=False)
        self.features = runner.results

        self.model_filename_json = VmafConfig.workspace_path("model", "test_save_load.json")

    def tearDown(self):
        if hasattr(self, 'model'):
            self.model.delete(self.model_filename_json, format='json')

    def test_train_save_load_predict_libsvmnusvr_json(self):

        xs = LibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = LibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = LibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = LibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename_json, format='json')
        self.assertTrue(os.path.exists(self.model_filename_json))
        self.assertFalse(os.path.exists(self.model_filename_json + '.model'))

        loaded_model = LibsvmNusvrTrainTestModel.from_file(
            self.model_filename_json, logger=None, format='json')

        result = self.model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.62263086620058783, places=4)

        # loaded model generates slight numerical difference
        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEqual(result['RMSE'], 0.62263139871631323, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
