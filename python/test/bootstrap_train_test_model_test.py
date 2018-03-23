import os
import unittest

from vmaf.config import VmafConfig
from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor
from vmaf.routine import read_dataset
from vmaf.tools.misc import import_python_file
from vmaf.core.train_test_model import BootstrapLibsvmNusvrTrainTestModel, \
    BootstrapSklearnRandomForestTrainTestModel, ResidueBootstrapLibsvmNusvrTrainTestModel, \
    ResidueBootstrapRandomForestTrainTestModel

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class BootstrapTrainTestModelTest(unittest.TestCase):

    def setUp(self):

        train_dataset_path = VmafConfig.test_resource_path('test_image_dataset_diffdim.py')
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

    def test_train_predict_bootstrap_libsvmnusvr(self):

        print "test bootstrap libsvmnusvr train and predict..."

        xs = BootstrapLibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapLibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapLibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type':'normalize'}, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.6226308662005912, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.6728233250049881, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.39781075130382537)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.3976996507329407)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.7546572275110091)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type':'clip_0to1'}, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.8424314108711487, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.8030845707458507, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3453036856929115)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.4988700247577427)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.743184600621924)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'custom_clip_0to1_map': {
                'Moment_noref_feature_1st_score': [0.0, 100.0],
              },
             }, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.846391627665469, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.8053305953883215, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3513542908476358)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.4776821071424475)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.7592266711979061)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type':'clip_minus1to1'}, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.8314352752340988, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.7834536370157068, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3791925369630857)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.393570819500396)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.7510205036835922)

        model = BootstrapLibsvmNusvrTrainTestModel(
            {'norm_type':'none'}, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.23294283650716543, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.17381660619722974, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3514635738854286)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.6100321778191451)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.6178566905304246)

    def test_train_save_load_predict_bootstrap_libsvmnusvr(self):

        print "test bootstrap libsvmnusvr train, save, load and predict..."

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

        self.assertAlmostEquals(self.model.evaluate(xs, ys)['RMSE'], 0.6226308662005912, places=4)
        self.assertAlmostEquals(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.6728233250049881, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.39781075130382537)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.3976996507329407)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.7546572275110091)

        self.assertAlmostEquals(loaded_model.evaluate(xs, ys)['RMSE'], 0.6226308662005912, places=4)
        self.assertAlmostEquals(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.6728233250049881, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.39781075130382537)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.3976996507329407)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.7546572275110091)

    def test_train_predict_bootstrap_randomforest(self):

        print "test bootstrap randomforest train and predict..."

        xs = BootstrapSklearnRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapSklearnRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapSklearnRandomForestTrainTestModel.get_xys_from_results(self.features)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type':'normalize', 'random_state':0}, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.1763473935351854, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.17259954003850736, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3516241931557006)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.416611111111111)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.6802222222222225)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type':'clip_0to1', 'random_state':0}, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.15634392282439022, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.16451548220306667, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3542010471809347)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.4059999999999997)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.668388888888888)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type': 'custom_clip_0to1',
             'custom_clip_0to1_map': {
                'Moment_noref_feature_1st_score': [0.0, 100.0],
              },
             'random_state': 0
             }, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.15634392282439022, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.16451548220306667, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3542010471809347)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.4059999999999997)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.668388888888888)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type':'clip_minus1to1', 'random_state':0}, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.15634392282439027, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.16559534633792203, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.3549729683560226)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.4026666666666663)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.6877222222222219)

        model = BootstrapSklearnRandomForestTrainTestModel(
            {'norm_type':'none', 'random_state':0}, None)
        model.train(xys)
        self.assertAlmostEquals(model.evaluate(xs, ys)['RMSE'], 0.15634392282438925, places=4)
        self.assertAlmostEquals(model.evaluate_bagging(xs, ys)['RMSE'], 0.16869299073806704, places=4)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_stddev'], 0.354125948341701)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_low'], 3.4002777777777777)
        self.assertAlmostEqual(model.evaluate_stddev(xs)['mean_ci95_high'], 4.6797222222222228)

    def test_train_save_load_predict_bootstrap_randomforest(self):

        print "test bootstrap randomforest train, save, load and predict..."

        xs = BootstrapSklearnRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = BootstrapSklearnRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = BootstrapSklearnRandomForestTrainTestModel.get_xys_from_results(self.features)

        self.model = BootstrapSklearnRandomForestTrainTestModel({'norm_type': 'normalize', 'random_state': 0}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)

        loaded_model = BootstrapSklearnRandomForestTrainTestModel.from_file(self.model_filename, None)

        self.assertAlmostEquals(self.model.evaluate(xs, ys)['RMSE'], 0.1763473935351854, places=4)
        self.assertAlmostEquals(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.17259954003850736, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.3516241931557006)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.416611111111111)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.6802222222222225)

        self.assertAlmostEquals(loaded_model.evaluate(xs, ys)['RMSE'], 0.1763473935351854, places=4)
        self.assertAlmostEquals(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.17259954003850736, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.3516241931557006)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.416611111111111)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.6802222222222225)

    def test_train_save_load_predict_residue_bootstrap_libsvmnusvr(self):

        print "test residue bootstrap libsvmnusvr train, save, load and predict..."

        xs = ResidueBootstrapLibsvmNusvrTrainTestModel.get_xs_from_results(self.features)
        ys = ResidueBootstrapLibsvmNusvrTrainTestModel.get_ys_from_results(self.features)
        xys = ResidueBootstrapLibsvmNusvrTrainTestModel.get_xys_from_results(self.features)

        self.model = ResidueBootstrapLibsvmNusvrTrainTestModel({'norm_type': 'normalize'}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)

        loaded_model = ResidueBootstrapLibsvmNusvrTrainTestModel.from_file(self.model_filename, None)

        self.assertAlmostEquals(self.model.evaluate(xs, ys)['RMSE'], 0.6226308662005912, places=4)
        self.assertAlmostEquals(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.9536184380768152, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.34425005186339697)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.466082152534141)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.6713510476530766)

        self.assertAlmostEquals(loaded_model.evaluate(xs, ys)['RMSE'], 0.6226308662005912, places=4)
        self.assertAlmostEquals(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.953618377381095, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.34425005186339697)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.466082152534141)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.6713510476530766)

    def test_train_save_load_predict_residue_bootstrap_randomforest(self):

        print "test residue bootstrap randomforest train, save, load and predict..."

        xs = ResidueBootstrapRandomForestTrainTestModel.get_xs_from_results(self.features)
        ys = ResidueBootstrapRandomForestTrainTestModel.get_ys_from_results(self.features)
        xys = ResidueBootstrapRandomForestTrainTestModel.get_xys_from_results(self.features)

        self.model = ResidueBootstrapRandomForestTrainTestModel({'norm_type': 'normalize', 'random_state': 0}, None)
        self.model.train(xys)

        self.model.to_file(self.model_filename)

        loaded_model = ResidueBootstrapRandomForestTrainTestModel.from_file(self.model_filename, None)

        self.assertAlmostEquals(self.model.evaluate(xs, ys)['RMSE'], 0.1763473935351854, places=4)
        self.assertAlmostEquals(self.model.evaluate_bagging(xs, ys)['RMSE'], 0.2771658143779744, places=4)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_stddev'], 0.2172301643431555)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.6215666666666664)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.4113055555555558)

        self.assertAlmostEquals(loaded_model.evaluate(xs, ys)['RMSE'], 0.1763473935351854, places=4)
        self.assertAlmostEquals(loaded_model.evaluate_bagging(xs, ys)['RMSE'], 0.2771658143779744, places=4)
        self.assertAlmostEqual(loaded_model.evaluate_stddev(xs)['mean_stddev'], 0.2172301643431555)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_low'], 3.6215666666666664)
        self.assertAlmostEqual(self.model.evaluate_stddev(xs)['mean_ci95_high'], 4.4113055555555558)

if __name__ == '__main__':
    unittest.main()
