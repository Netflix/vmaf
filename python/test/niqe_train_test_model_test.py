__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import unittest
import os

import numpy as np

from vmaf.config import VmafConfig
from vmaf.core.train_test_model import TrainTestModel
from vmaf.routine import read_dataset
from vmaf.tools.misc import import_python_file, empty_object
from vmaf.core.noref_feature_extractor import NiqeNorefFeatureExtractor
from vmaf.core.niqe_train_test_model import NiqeTrainTestModel


class NiqeTrainTestModelTest(unittest.TestCase):

    def setUp(self):

        train_dataset_path = VmafConfig.test_resource_path('test_image_dataset.py')
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        param = empty_object()
        param.model_type = "NIQE"
        param.model_param_dict = {
            'patch_size': 96,
        }
        self.param = param

        optional_dict = {'mode': 'train'}

        self.fextractor = NiqeNorefFeatureExtractor(
            train_assets,
            None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict=optional_dict,
            optional_dict2=None,
        )
        self.fextractor.run(parallelize=True)
        self.features = self.fextractor.results

        self.model_filename = VmafConfig.workspace_path('model', 'test_save_load.pkl')

    def tearDown(self):
        if os.path.exists(self.model_filename):
            os.remove(self.model_filename)

    def test_get_xs_from_results(self):
        xs = NiqeTrainTestModel.get_xs_from_results(self.features)

        self.assertEqual(len(xs['NIQE_noref_feature_N11_scores'][0]), 5)
        self.assertEqual(len(xs['NIQE_noref_feature_N11_scores'][1]), 3)
        self.assertEqual(len(xs['NIQE_noref_feature_N11_scores'][2]), 7)
        self.assertEqual(len(xs['NIQE_noref_feature_N11_scores'][3]), 11)
        self.assertEqual(len(xs['NIQE_noref_feature_N11_scores'][4]), 3)
        self.assertAlmostEqual(xs['NIQE_noref_feature_N11_scores'][0][3],
                               -0.016672410493636325)

    def test_train(self):
        xys = NiqeTrainTestModel.get_xys_from_results(self.features)
        model = NiqeTrainTestModel(self.param.model_param_dict, None)
        model.train(xys)
        self.assertAlmostEqual(np.mean(model.model_dict['model']['mu']), 0.58721456594247923, places=4)
        self.assertAlmostEqual(np.mean(model.model_dict['model']['cov']), 0.0062869078795764156, places=4)

    def test_predict(self):
        xs = NiqeTrainTestModel.get_xs_from_results(self.features)
        xys = NiqeTrainTestModel.get_xys_from_results(self.features)
        model = NiqeTrainTestModel(self.param.model_param_dict, None)
        model.train(xys)
        ys_pred = model.predict(xs)['ys_label_pred']
        self.assertAlmostEqual(np.mean(ys_pred), 3.7645955838278695, places=4)
        self.assertAlmostEqual(ys_pred[0], 3.5840187532885257, places=3)
        self.assertAlmostEqual(ys_pred[1], 5.1472584219091102, places=4)

    def test_save_load_model(self):
        xs = NiqeTrainTestModel.get_xs_from_results(self.features)
        xys = NiqeTrainTestModel.get_xys_from_results(self.features)
        model = NiqeTrainTestModel(self.param.model_param_dict, None)
        model.train(xys)
        model.to_file(self.model_filename)
        model_new = TrainTestModel.from_file(self.model_filename, None)
        ys_pred = model_new.predict(xs)['ys_label_pred']
        self.assertAlmostEqual(np.mean(ys_pred), 3.7645955838278695, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
