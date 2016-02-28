__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import os
import unittest
import config
from run_testing import read_dataset
from run_training import train_on_dataset
from tools import import_python_file


class TestTrainOnDataset(unittest.TestCase):

    def setUp(self):
        self.train_dataset = import_python_file(config.ROOT + '/python/test/resource/dataset_sample.py')
        self.output_model_filepath = config.ROOT + "/workspace/model/test_output_model.pkl"

    def tearDown(self):
        if hasattr(self, 'train_fassembler'):
            self.train_fassembler.remove_logs()
        if os.path.exists(self.output_model_filepath): os.remove(self.output_model_filepath)

    def test_train_on_dataset_with_dis1st_thr(self):
        feature_param = import_python_file(config.ROOT + '/python/test/resource/feature_param_sample.py')
        model_param = import_python_file(config.ROOT + '/python/teset/resource/model_param_sample.py')
        train_fassembler, train_ys_pred = \
            train_on_dataset(train_dataset=self.train_dataset,
                         feature_param=feature_param,
                         model_param=model_param,
                         ax=None,
                         result_store=None,
                         output_model_filepath=self.output_model_filepath,
                         parallelize=True,
                         fifo_mode=True,
                         logger=None,
                         )
        self.train_fassembler = train_fassembler
        self.assertTrue(os.path.exists(self.output_model_filepath))
        self.assertAlmostEquals(train_ys_pred,
                                [91.707522376672316, 58.277822562766268,
                                 95.577344864265129, 90.255945341860439])



if __name__ == '__main__':
    unittest.main()
