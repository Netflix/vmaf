__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import os
import unittest

import config
from routine import train_test_on_dataset
from tools.misc import import_python_file


class TestTrainOnDataset(unittest.TestCase):

    def setUp(self):
        self.train_dataset = import_python_file(
            config.ROOT + '/python/test/resource/dataset_sample.py')
        self.output_model_filepath = \
            config.ROOT + "/workspace/model/test_output_model.pkl"

    def tearDown(self):
        if os.path.exists(self.output_model_filepath):
            os.remove(self.output_model_filepath)

    def test_train_test_on_dataset_with_dis1st_thr(self):
        model_param = import_python_file(
            config.ROOT + '/python/test/resource/model_param_sample.py')
        feature_param = import_python_file(
            config.ROOT + '/python/test/resource/feature_param_sample.py')
        train_fassembler, train_assets, train_stats, \
        test_fassembler, test_assets, test_stats = \
            train_test_on_dataset(
                train_dataset=self.train_dataset, test_dataset=None,
                         feature_param=feature_param, model_param=model_param,
                         train_ax=None, test_ax=None, result_store=None,
                         parallelize=True,
                         logger=None,
                         fifo_mode=True,
                         output_model_filepath=self.output_model_filepath,
                         )
        self.train_fassembler = train_fassembler
        self.assertTrue(os.path.exists(self.output_model_filepath))
        self.assertItemsEqual(train_stats['ys_label_pred'],
                                [91.707522376672316, 58.277822562766268,
                                 91.707521620497104, 88.307134410232536])

if __name__ == '__main__':
    unittest.main()
