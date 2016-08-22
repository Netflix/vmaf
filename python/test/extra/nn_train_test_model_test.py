import os
import unittest

import numpy as np

import config
from core.executor import run_executors_in_parallel
from core.nn_train_test_model import NeuralNetTrainTestModel, \
    ToddNoiseClassifierTrainTestModel
from core.raw_extractor import DisYUVRawVideoExtractor
from routine import read_dataset
from tools.misc import import_python_file

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class ToddNoiseClassificationExampleTest(unittest.TestCase):

    def setUp(self):

        train_dataset_path = config.ROOT + '/python/test/resource/test_image_dataset_noisy.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.raw_video_h5py_filepath = config.ROOT + '/workspace/workdir/rawvideo.hdf5'
        self.raw_video_h5py_file = DisYUVRawVideoExtractor.open_h5py_file(
            self.raw_video_h5py_filepath)
        optional_dict2 = {'h5py_file': self.raw_video_h5py_file}

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

        np.random.seed(0)
        np.random.shuffle(self.features)

        self.patch_h5py_filepath = config.ROOT + '/workspace/workdir/patch.hdf5'

        self.model_filename = config.ROOT + "/workspace/model/test_save_load.pkl"

        NeuralNetTrainTestModel.reset()

    def tearDown(self):
        if hasattr(self, 'raw_video_h5py_file'):
            DisYUVRawVideoExtractor.close_h5py_file(self.raw_video_h5py_file)
        if os.path.exists(self.raw_video_h5py_filepath):
            os.remove(self.raw_video_h5py_filepath)
        if os.path.exists(self.patch_h5py_filepath):
            os.remove(self.patch_h5py_filepath)
        if hasattr(self, 'model'):
            self.model.delete(self.model_filename)

    def test_extracted_features(self):
        self.assertAlmostEqual(np.mean(self.features[0]['dis_y']), 126.53109111987617, places=4)
        self.assertAlmostEqual(np.mean(self.features[9]['dis_y']), 160.17343151922591, places=4)
        self.assertAlmostEqual(np.mean(self.features[0]['dis_u']), 120.98193016884606, places=4)
        self.assertAlmostEqual(np.mean(self.features[9]['dis_u']), 134.88002020712301, places=4)
        self.assertAlmostEqual(np.mean(self.features[0]['dis_v']), 124.04507095161301, places=4)
        self.assertAlmostEqual(np.mean(self.features[9]['dis_v']), 124.80777974235919, places=4)

    def test_train_predict(self):

        xys = ToddNoiseClassifierTrainTestModel.get_xys_from_results(self.features[:9])
        xs = ToddNoiseClassifierTrainTestModel.get_xs_from_results(self.features[9:])
        ys = ToddNoiseClassifierTrainTestModel.get_ys_from_results(self.features[9:])

        patch_h5py_file = ToddNoiseClassifierTrainTestModel.open_h5py_file(self.patch_h5py_filepath)
        optional_dict2 = {'h5py_file': patch_h5py_file}

        self.model = ToddNoiseClassifierTrainTestModel(
            param_dict={'seed': 0, 'n_epochs': 5},
            optional_dict2=optional_dict2)
        self.model.train(xys)

        result = self.model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.66666666666666663, places=4)
        self.assertAlmostEquals(result['f1'], 0.7142857142857143, places=4)
        self.assertAlmostEquals(result['errorrate'], 0.44444444444444442, places=4)

        ToddNoiseClassifierTrainTestModel.close_h5py_file(patch_h5py_file)

    def test_train_save_load_predict(self):

        xys = ToddNoiseClassifierTrainTestModel.get_xys_from_results(self.features[:9])
        xs = ToddNoiseClassifierTrainTestModel.get_xs_from_results(self.features[9:])
        ys = ToddNoiseClassifierTrainTestModel.get_ys_from_results(self.features[9:])

        patch_h5py_file = ToddNoiseClassifierTrainTestModel.open_h5py_file(self.patch_h5py_filepath)
        param_dict = {'seed': 0, 'n_epochs': 5}
        optional_dict2 = {'h5py_file': patch_h5py_file}

        self.model = ToddNoiseClassifierTrainTestModel(
            param_dict=param_dict,
            optional_dict2=optional_dict2)
        self.model.train(xys)

        self.model.to_file(self.model_filename)

        # before loading model, clean up any memory first
        NeuralNetTrainTestModel.reset()

        loaded_model = ToddNoiseClassifierTrainTestModel.from_file(
            self.model_filename,
            optional_dict2=optional_dict2)

        result = loaded_model.evaluate(xs, ys)
        self.assertAlmostEquals(result['RMSE'], 0.66666666666666663, places=4)
        self.assertAlmostEquals(result['f1'], 0.7142857142857143, places=4)
        self.assertAlmostEquals(result['errorrate'], 0.44444444444444442, places=4)

        ToddNoiseClassifierTrainTestModel.close_h5py_file(patch_h5py_file)