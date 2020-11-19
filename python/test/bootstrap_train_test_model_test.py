import os
import unittest
import numpy as np

from vmaf.config import VmafConfig
from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor
from vmaf.routine import read_dataset
from vmaf.tools.misc import import_python_file
from vmaf.core.train_test_model import BootstrapLibsvmNusvrTrainTestModel, \
    BootstrapSklearnRandomForestTrainTestModel, ResidueBootstrapLibsvmNusvrTrainTestModel, \
    ResidueBootstrapRandomForestTrainTestModel

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class BootstrapTrainTestModelTest(unittest.TestCase):

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

        self.model_filename = VmafConfig.workspace_path("model", "test_save_load.pkl")

    def tearDown(self):
        if hasattr(self, 'model'):
            self.model.delete(self.model_filename)

    def test_train_predict_bootstrap_libsvmnusvr(self):

        xs = BootstrapLibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapLibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapLibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type': 'normalize'}, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.6226308662005923, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.6696474832672723, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.39703824367678103, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.3984211902006627, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.753825468723706, places=2)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type': 'clip_0to1'}, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.8424314108711469, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.8035115098260148, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.34509843867201334, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.5003674768667943, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.742621319805588, places=2)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'custom_clip_0to1_map': {
                'Moment_noref_feature_1st_score': [0.0, 100.0],
              },
             }, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.8463916276654698, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.8056949768422423, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3510958063355727, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.479040231431352, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.758095707489271, places=2)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type': 'clip_minus1to1'}, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.831435275234099, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.7833205217865182, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.37890490520445996, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.395027628418071, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.750492595014303, places=2)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type': 'none'}, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.23294283650716543, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.17423420271622292, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3505327957559016, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.6104498263465024, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.617611259682587, places=2)

    def test_train_save_load_predict_bootstrap_libsvmnusvr(self):

        xs = BootstrapLibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapLibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapLibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = BootstrapLibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)

        self.assertTrue(os.path.exists('{}'.format(self.model_filename)))
        self.assertTrue(os.path.exists('{}'.format(self.model_filename) + '.model'))
        for i in range(1, 100):
            self.assertTrue(os.path.exists('{}.{:04d}'.format(self.model_filename, i)))
            self.assertTrue(os.path.exists('{}.{:04d}'.format(self.model_filename, i) + '.model'))

        loaded_model = BootstrapLibsvmNusvrTrainTestModel.from_file(self.model_filename, None)

        self.assertAlmostEqual(self.model.evaluate(xs, ys)['RMSE'], 0.6226308662005923, places=4)
        self.assertAlmostEqual(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.6696474832672723, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.39703824367678103, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.3984211902006627, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.753825468723706, places=2)

        # loaded model generates slight numerical difference
        self.assertAlmostEqual(loaded_model.evaluate(xs, ys)['RMSE'], 0.6226313987163097, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.6696478863129723, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.3970382109813205, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_low'], 3.398421319902801, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_high'], 4.753825575324514, places=2)

    def test_train_across_model_stats_bootstraplibsvmnusvr(self):

        xs = BootstrapLibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapLibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapLibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = BootstrapLibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)
        self.model.to_file(self.model_filename)

        model_predictions = self.model.predict(xs)
        ys_label_pred = model_predictions['ys_label_pred']
        ys_label_pred_all_models = model_predictions['ys_label_pred_all_models']
        ys_label = ys['label']
        stats = self.model.get_stats(ys_label, ys_label_pred,
                                     ys_label_pred_all_models=ys_label_pred_all_models)

        # check that across model stats are generated
        assert 'SRCC_across_model_distribution' in stats \
               and 'PCC_across_model_distribution' in stats \
               and 'RMSE_across_model_distribution' in stats

        # check dimensions
        assert len(stats['SRCC_across_model_distribution']) == np.shape(ys_label_pred_all_models)[0] \
            and len(stats['PCC_across_model_distribution']) == np.shape(ys_label_pred_all_models)[0] \
            and len(stats['RMSE_across_model_distribution']) == np.shape(ys_label_pred_all_models)[0]

        # check case without across_model_stats
        stats_not_across_model = self.model.get_stats(ys_label, ys_label_pred)

        assert 'SRCC_across_model_distribution' not in stats_not_across_model \
               and 'PCC_across_model_distribution' not in stats_not_across_model \
               and 'RMSE_across_model_distribution' not in stats_not_across_model

    def test_train_predict_bootstrap_randomforest(self):

        xs = BootstrapSklearnRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapSklearnRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapSklearnRandomForestTrainTestModel.get_xys_from_results(self.features)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type': 'normalize',
             'n_estimators': 10,
             'random_state': 0}, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.17634739353518517, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.1734023350765026, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.35136642421562986, places=3)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.4185833333333333, places=1)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.6789999999999985, places=1)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type': 'clip_0to1',
             'n_estimators': 10,
             'random_state': 0}, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.15634392282439, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.16538552470948875, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3539523612820203, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.4091111111111108, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.667805555555555, places=1)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'n_estimators': 10,
             'custom_clip_0to1_map': {
                'Moment_noref_feature_1st_score': [0.0, 100.0],
              },
             'random_state': 0
             }, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.15634392282439, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.16538552470948875, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3539523612820203, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.4091111111111108, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.667805555555555, places=1)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type': 'clip_minus1to1',
             'n_estimators': 10,
             'random_state': 0}, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.15634392282438947, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.16634325875304712, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.35470969070245834, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.405222222222222, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.686916666666666, places=2)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type': 'none',
             'n_estimators': 10,
             'random_state': 0}, None)
        model.train(xys)
        self.assertAlmostEqual(model.evaluate(xs, ys)['RMSE'], 0.15634392282438941, places=4)
        self.assertAlmostEqual(model.evaluate_bagging(xs, ys)['RMSE'], 0.16935240638616547, places=3)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3538762838023098, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.402638888888889, places=2)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.679027777777778, places=2)

    def test_train_save_load_predict_bootstrap_randomforest(self):

        xs = BootstrapSklearnRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapSklearnRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapSklearnRandomForestTrainTestModel.get_xys_from_results(self.features)

        self.model = BootstrapSklearnRandomForestTrainTestModel({'norm_type': 'normalize',
                                                                 'n_estimators': 10,
                                                                 'random_state': 0}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)

        loaded_model = BootstrapSklearnRandomForestTrainTestModel.from_file(self.model_filename, None)

        self.assertAlmostEqual(self.model.evaluate(xs, ys)['RMSE'], 0.17634739353518517, places=4)
        self.assertAlmostEqual(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.1734023350765026, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.35136642421562986, places=3)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.4185833333333333, places=1)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.6789999999999985, places=1)

        self.assertAlmostEqual(loaded_model.evaluate(xs, ys)['RMSE'], 0.17634739353518517, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.1734023350765026, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.35136642421562986, places=3)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_low'], 3.4185833333333333, places=1)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_high'], 4.6789999999999985, places=1)

    def test_train_save_load_predict_residue_bootstrap_libsvmnusvr(self):

        xs = ResidueBootstrapLibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = ResidueBootstrapLibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = ResidueBootstrapLibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = ResidueBootstrapLibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)

        loaded_model = ResidueBootstrapLibsvmNusvrTrainTestModel.from_file(self.model_filename, None)

        self.assertAlmostEqual(self.model.evaluate(xs, ys)['RMSE'], 0.6226308662005923, places=4)
        self.assertAlmostEqual(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.955069756575906, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.3437214386264522, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.4674811219156174, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.671073568422049, places=2)

        self.assertAlmostEqual(loaded_model.evaluate(xs, ys)['RMSE'], 0.6226313987163097, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.9550696941149013, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.3437213993262196, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_low'], 3.4674811883034935, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_high'], 4.671073598975557, places=2)

    def test_train_save_load_predict_residue_bootstrap_randomforest(self):

        xs = ResidueBootstrapRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = ResidueBootstrapRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = ResidueBootstrapRandomForestTrainTestModel.get_xys_from_results(self.features)

        self.model = ResidueBootstrapRandomForestTrainTestModel({'norm_type': 'normalize',
                                                                 'n_estimators': 10,
                                                                 'random_state': 0}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)

        loaded_model = ResidueBootstrapRandomForestTrainTestModel.from_file(self.model_filename, None)

        self.assertAlmostEqual(self.model.evaluate(xs, ys)['RMSE'], 0.17634739353518517, places=4)
        self.assertAlmostEqual(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.2731966696149454, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.2176592325445535, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.621466666666666, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.4107916666666656, places=2)

        self.assertAlmostEqual(loaded_model.evaluate(xs, ys)['RMSE'], 0.17634739353518517, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.2731966696149454, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.2176592325445535, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_low'], 3.621466666666666, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_high'], 4.4107916666666656, places=2)


class BootstrapTrainTestModelTestJson(unittest.TestCase):

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

    def test_train_save_load_predict_bootstrap_libsvmnusvr_json(self):

        xs = BootstrapLibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapLibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapLibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = BootstrapLibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename_json, format='json')

        self.assertTrue(os.path.exists('{}'.format(self.model_filename_json)))
        self.assertFalse(os.path.exists('{}'.format(self.model_filename_json) + '.model'))
        for i in range(1, 100):
            self.assertTrue(os.path.exists('{}.{:04d}'.format(self.model_filename_json, i)))
            self.assertFalse(os.path.exists('{}.{:04d}'.format(self.model_filename_json, i) + '.model'))

        loaded_model = BootstrapLibsvmNusvrTrainTestModel.from_file(self.model_filename_json, None, format='json')

        self.assertAlmostEqual(self.model.evaluate(xs, ys)['RMSE'], 0.6226308662005923, places=4)
        self.assertAlmostEqual(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.6696474832672723, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.39703824367678103, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.3984211902006627, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.753825468723706, places=2)

        # loaded model generates slight numerical difference
        self.assertAlmostEqual(loaded_model.evaluate(xs, ys)['RMSE'], 0.6226313987163097, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.6696478863129723, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.3970382109813205, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_low'], 3.398421319902801, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_high'], 4.753825575324514, places=2)


class BootstrapTrainTestModelTestPklCombined(unittest.TestCase):

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

        self.model_filename_pkl = VmafConfig.workspace_path("model", "test_save_load.pkl")

    def tearDown(self):
        pass

    def test_train_save_load_predict_bootstrap_libsvmnusvr_pkl_combined(self):
        xys = BootstrapLibsvmNusvrTrainTestModel.get_xys_from_results(self.features)
        self.model = BootstrapLibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)
        with self.assertRaises(AssertionError):
            self.model.to_file(self.model_filename_pkl, format='pkl', combined=True)


class BootstrapTrainTestModelTestJsonCombined(unittest.TestCase):

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
            self.model.delete(self.model_filename_json, format='json', combined=True)

    def test_train_save_load_predict_bootstrap_libsvmnusvr_json_combined(self):

        xs = BootstrapLibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapLibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapLibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = BootstrapLibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename_json, format='json', combined=True)

        self.assertTrue(os.path.exists('{}'.format(self.model_filename_json)))
        self.assertFalse(os.path.exists('{}'.format(self.model_filename_json) + '.model'))
        for i in range(1, 100):
            self.assertFalse(os.path.exists('{}.{:04d}'.format(self.model_filename_json, i)))
            self.assertFalse(os.path.exists('{}.{:04d}'.format(self.model_filename_json, i) + '.model'))

        loaded_model = BootstrapLibsvmNusvrTrainTestModel.from_file(self.model_filename_json, None, format='json', combined=True)

        self.assertAlmostEqual(self.model.evaluate(xs, ys)['RMSE'], 0.6226308662005923, places=4)
        self.assertAlmostEqual(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.6696474832672723, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.39703824367678103, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.3984211902006627, places=2)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.753825468723706, places=2)

        # loaded model generates slight numerical difference
        self.assertAlmostEqual(loaded_model.evaluate(xs, ys)['RMSE'], 0.6226313987163097, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.6696478863129723, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.3970382109813205, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_low'], 3.398421319902801, places=2)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_ci95_high'], 4.753825575324514, places=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
