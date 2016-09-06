__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import unittest

import config
from routine import train_test_vmaf_on_dataset, read_dataset
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
        test_fassembler, test_assets, test_stats, _ = \
            train_test_vmaf_on_dataset(
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
                                [90.753010402770798, 59.223801498461015,
                                 90.753011435798058, 89.270176556597008])

class TestReadDataset(unittest.TestCase):

    def test_read_dataset(self):
        train_dataset_path = config.ROOT + '/python/test/resource/test_image_dataset_diffdim.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertEquals(len(train_assets), 9)
        self.assertTrue('groundtruth' in train_assets[0].asset_dict.keys())
        self.assertTrue('os' not in train_assets[0].asset_dict.keys())
        self.assertTrue('width' in train_assets[0].asset_dict.keys())
        self.assertTrue('height' in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_width' not in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_height' not in train_assets[0].asset_dict.keys())

    def test_read_dataset_qualitywh(self):
        train_dataset_path = config.ROOT + '/python/test/resource/test_image_dataset_diffdim_qualitywh.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertTrue('quality_width' in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_height' in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_width' in train_assets[1].asset_dict.keys())
        self.assertTrue('quality_height' in train_assets[1].asset_dict.keys())
        self.assertEqual(train_assets[0].asset_dict['quality_width'], 200)
        self.assertEqual(train_assets[0].asset_dict['quality_height'], 100)
        self.assertEqual(train_assets[1].asset_dict['quality_width'], 200)
        self.assertEqual(train_assets[1].asset_dict['quality_height'], 100)

    def test_read_dataset_qualitywh2(self):
        train_dataset_path = config.ROOT + '/python/test/resource/test_image_dataset_diffdim_qualitywh2.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertTrue('quality_width' in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_height' in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_width' not in train_assets[1].asset_dict.keys())
        self.assertTrue('quality_height' not in train_assets[1].asset_dict.keys())
        self.assertEqual(train_assets[0].asset_dict['quality_width'], 200)
        self.assertEqual(train_assets[0].asset_dict['quality_height'], 100)

if __name__ == '__main__':
    unittest.main()
