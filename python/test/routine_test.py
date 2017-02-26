import os
import unittest

from vmaf import config
from vmaf.routine import train_test_vmaf_on_dataset, read_dataset, run_test_on_dataset, generate_dataset_from_raw
from vmaf.tools.misc import import_python_file
from vmaf.core.quality_runner import VmafQualityRunner
from vmaf.mos.subjective_model import MosModel

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"


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
        self.assertTrue('resampling_type' in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_width' in train_assets[1].asset_dict.keys())
        self.assertTrue('quality_height' in train_assets[1].asset_dict.keys())
        self.assertTrue('resampling_type' in train_assets[1].asset_dict.keys())
        self.assertEqual(train_assets[0].asset_dict['quality_width'], 200)
        self.assertEqual(train_assets[0].asset_dict['quality_height'], 100)
        self.assertEqual(train_assets[0].asset_dict['resampling_type'], 'bicubic')
        self.assertEqual(train_assets[1].asset_dict['quality_width'], 200)
        self.assertEqual(train_assets[1].asset_dict['quality_height'], 100)
        self.assertEqual(train_assets[1].asset_dict['resampling_type'], 'bicubic')

    def test_read_dataset_qualitywh2(self):
        train_dataset_path = config.ROOT + '/python/test/resource/test_image_dataset_diffdim_qualitywh2.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertTrue('quality_width' in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_height' in train_assets[0].asset_dict.keys())
        self.assertTrue('resampling_type' in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_width' not in train_assets[1].asset_dict.keys())
        self.assertTrue('quality_height' not in train_assets[1].asset_dict.keys())
        self.assertTrue('resampling_type' not in train_assets[1].asset_dict.keys())
        self.assertEqual(train_assets[0].asset_dict['quality_width'], 200)
        self.assertEqual(train_assets[0].asset_dict['quality_height'], 100)
        self.assertEqual(train_assets[0].asset_dict['resampling_type'], 'bicubic')

    def test_read_dataset_diffyuv(self):
        train_dataset_path = config.ROOT + '/python/test/resource/test_dataset_diffyuv.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertEquals(len(train_assets), 4)
        self.assertEquals(train_assets[0].ref_width_height, (1920, 1080))
        self.assertEquals(train_assets[0].dis_width_height, (1920, 1080))
        self.assertEquals(train_assets[0].quality_width_height, (1920, 1080))
        self.assertEquals(train_assets[0].yuv_type, 'yuv420p')
        self.assertEquals(train_assets[2].ref_width_height, (1280, 720))
        self.assertEquals(train_assets[2].dis_width_height, (1280, 720))
        self.assertEquals(train_assets[2].quality_width_height, (1280, 720))
        self.assertEquals(train_assets[2].yuv_type, 'yuv420p10le')

class TestTrainOnDataset(unittest.TestCase):

    def setUp(self):
        self.output_model_filepath = \
            config.ROOT + "/workspace/model/test_output_model.pkl"

    def tearDown(self):
        if os.path.exists(self.output_model_filepath):
            os.remove(self.output_model_filepath)

    def test_train_test_on_dataset_with_dis1st_thr(self):
        train_dataset = import_python_file(
            config.ROOT + '/python/test/resource/dataset_sample.py')
        model_param = import_python_file(
            config.ROOT + '/python/test/resource/model_param_sample.py')
        feature_param = import_python_file(
            config.ROOT + '/python/test/resource/feature_param_sample.py')

        train_fassembler, train_assets, train_stats, test_fassembler, test_assets, test_stats, _ = train_test_vmaf_on_dataset(
            train_dataset=train_dataset,
            test_dataset=train_dataset,
            feature_param=feature_param,
            model_param=model_param,
            train_ax=None,
            test_ax=None,
            result_store=None,
            parallelize=True,
            logger=None,
            fifo_mode=True,
            output_model_filepath=self.output_model_filepath,
        )

        self.train_fassembler = train_fassembler
        self.assertTrue(os.path.exists(self.output_model_filepath))
        self.assertAlmostEqual(train_stats['ys_label_pred'][0], 90.753010402770798, places=3)
        self.assertAlmostEqual(test_stats['ys_label_pred'][0], 90.753010402770798, places=3)

    def test_train_test_on_raw_dataset_with_dis1st_thr(self):
        train_dataset = import_python_file(
            config.ROOT + '/python/test/resource/raw_dataset_sample.py')
        model_param = import_python_file(
            config.ROOT + '/python/test/resource/model_param_sample.py')
        feature_param = import_python_file(
            config.ROOT + '/python/test/resource/feature_param_sample.py')

        train_fassembler, train_assets, train_stats, test_fassembler, test_assets, test_stats, _ = train_test_vmaf_on_dataset(
            train_dataset=train_dataset,
            test_dataset=train_dataset,
            feature_param=feature_param,
            model_param=model_param,
            train_ax=None,
            test_ax=None,
            result_store=None,
            parallelize=True,
            logger=None,
            fifo_mode=True,
            output_model_filepath=self.output_model_filepath
        )

        self.train_fassembler = train_fassembler
        self.assertTrue(os.path.exists(self.output_model_filepath))
        self.assertAlmostEqual(train_stats['ys_label_pred'][0], 93.565459224020742, places=3)
        self.assertAlmostEqual(test_stats['ys_label_pred'][0], 93.565459224020742, places=3)

    def test_test_on_dataset(self):
        test_dataset = import_python_file(
            config.ROOT + '/python/test/resource/dataset_sample.py')
        test_assets, results = run_test_on_dataset(test_dataset, VmafQualityRunner, None,
                        None, None,
                        parallelize=True,
                        aggregate_method=None)

        self.assertAlmostEqual(results[0]['VMAF_score'], 99.142659046424384, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 35.066157497128764, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 97.428042675471147, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 97.427927701008869, places=4)
        self.assertAlmostEqual(test_assets[0].groundtruth, 100, places=4)
        self.assertAlmostEqual(test_assets[1].groundtruth, 50, places=4)
        self.assertAlmostEqual(test_assets[2].groundtruth, 100, places=4)
        self.assertAlmostEqual(test_assets[3].groundtruth, 80, places=4)

    def test_test_on_dataset_raw(self):
        test_dataset = import_python_file(
            config.ROOT + '/python/test/resource/raw_dataset_sample.py')
        test_assets, results = run_test_on_dataset(test_dataset, VmafQualityRunner, None,
                        None, None,
                        parallelize=True,
                        aggregate_method=None)

        self.assertAlmostEqual(results[0]['VMAF_score'], 99.142659046424384, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 35.066157497128764, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 97.428042675471147, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 97.427927701008869, places=4)
        self.assertAlmostEqual(test_assets[0].groundtruth, 100, places=4)
        self.assertAlmostEqual(test_assets[1].groundtruth, 50, places=4)
        self.assertAlmostEqual(test_assets[2].groundtruth, 100, places=4)
        self.assertAlmostEqual(test_assets[3].groundtruth, 90, places=4)

    def test_test_on_dataset_mle(self):
        test_dataset = import_python_file(
            config.ROOT + '/python/test/resource/raw_dataset_sample.py')
        test_assets, results = run_test_on_dataset(test_dataset, VmafQualityRunner, None,
                        None, None,
                        parallelize=True,
                        aggregate_method=None,
                        subj_model_class=MosModel)

        self.assertAlmostEqual(results[0]['VMAF_score'], 99.142659046424384, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 35.066157497128764, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 97.428042675471147, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 97.427927701008869, places=4)
        self.assertAlmostEqual(test_assets[0].groundtruth, 100, places=4)
        self.assertAlmostEqual(test_assets[1].groundtruth, 50, places=4)
        self.assertAlmostEqual(test_assets[2].groundtruth, 90, places=4)
        self.assertAlmostEqual(test_assets[3].groundtruth, 80, places=4)

class TestGenerateDatasetFromRaw(unittest.TestCase):

    def setUp(self):
        self.raw_dataset_filepath = config.ROOT + '/resource/dataset/NFLX_dataset_public_raw.py'
        self.derived_dataset_path = config.ROOT + '/workspace/workdir/test_derived_dataset.py'
        self.derived_dataset_path_pyc = config.ROOT + '/workspace/workdir/test_derived_dataset.pyc'

    def tearDown(self):
        if os.path.exists(self.derived_dataset_path):
            os.remove(self.derived_dataset_path)
        if os.path.exists(self.derived_dataset_path_pyc):
            os.remove(self.derived_dataset_path_pyc)

    def test_generate_dataset_from_raw_default(self): # DMOS
        generate_dataset_from_raw(raw_dataset_filepath=self.raw_dataset_filepath,
                         output_dataset_filepath=self.derived_dataset_path)
        dataset = import_python_file(self.derived_dataset_path)
        self.assertAlmostEqual(dataset.dis_videos[0]['groundtruth'], 1.42307692308, places=4)

    def test_generate_dataset_from_raw_mos(self):
        generate_dataset_from_raw(raw_dataset_filepath=self.raw_dataset_filepath,
                                  output_dataset_filepath=self.derived_dataset_path,
                                  subj_model_class=MosModel)
        dataset = import_python_file(self.derived_dataset_path)
        self.assertAlmostEqual(dataset.dis_videos[0]['groundtruth'], 1.3076923076923077, places=4)

if __name__ == '__main__':
    unittest.main()
