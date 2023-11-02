import glob
import os
import unittest
import shutil

import numpy as np

from vmaf.config import VmafConfig, DisplayConfig
# from vmaf.routine import train_test_vmaf_on_dataset, read_dataset, run_test_on_dataset, generate_dataset_from_raw
from vmaf.routine import read_dataset, generate_dataset_from_raw, compare_two_quality_runners_on_dataset
from vmaf.tools.misc import import_python_file
from vmaf.core.quality_runner import VmafQualityRunner, BootstrapVmafQualityRunner, PsnrQualityRunner
from sureal.subjective_model import MosModel, SubjectiveModel

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class TestReadDataset(unittest.TestCase):

    def test_read_dataset(self):
        train_dataset_path = VmafConfig.test_resource_path('test_image_dataset_diffdim.py')
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertEqual(len(train_assets), 9)
        self.assertTrue('groundtruth' in train_assets[0].asset_dict.keys())
        self.assertTrue('os' not in train_assets[0].asset_dict.keys())
        self.assertFalse('width' in train_assets[0].asset_dict.keys())
        self.assertTrue('ref_width' in train_assets[0].asset_dict.keys())
        self.assertTrue('dis_width' in train_assets[0].asset_dict.keys())
        self.assertFalse('height' in train_assets[0].asset_dict.keys())
        self.assertTrue('ref_height' in train_assets[0].asset_dict.keys())
        self.assertTrue('dis_height' in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_width' not in train_assets[0].asset_dict.keys())
        self.assertTrue('quality_height' not in train_assets[0].asset_dict.keys())

    def test_read_dataset_qualitywh(self):
        train_dataset_path = VmafConfig.test_resource_path('test_image_dataset_diffdim_qualitywh.py')
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
        train_dataset_path = VmafConfig.test_resource_path('test_image_dataset_diffdim_qualitywh2.py')
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

    def test_read_dataset_fps_rebuf_indices(self):
        train_dataset_path = VmafConfig.test_resource_path('test_dataset_fps_rebufinds.py')
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertTrue('fps' in train_assets[0].asset_dict.keys())
        self.assertTrue('rebuf_indices' in train_assets[0].asset_dict.keys())

    def test_read_dataset_bad_fps_rebuf_indices(self):
        train_dataset_path = VmafConfig.test_resource_path('test_dataset_bad_fps_rebufinds.py')
        train_dataset = import_python_file(train_dataset_path)

        with self.assertRaises(AssertionError):
            train_assets = read_dataset(train_dataset)

    def test_read_dataset_fps_bad_rebuf_indices(self):
        train_dataset_path = VmafConfig.test_resource_path('test_dataset_fps_bad_rebufinds.py')
        train_dataset = import_python_file(train_dataset_path)

        with self.assertRaises(AssertionError):
            train_assets = read_dataset(train_dataset)

    def test_read_dataset_diffyuv(self):
        train_dataset_path = VmafConfig.test_resource_path('test_dataset_diffyuv.py')
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertEqual(len(train_assets), 4)
        self.assertEqual(train_assets[0].ref_width_height, (1920, 1080))
        self.assertEqual(train_assets[0].dis_width_height, (1920, 1080))
        self.assertEqual(train_assets[0].quality_width_height, (1920, 1080))
        self.assertEqual(train_assets[0].dis_yuv_type, 'yuv420p')
        self.assertEqual(train_assets[2].ref_width_height, (1280, 720))
        self.assertEqual(train_assets[2].dis_width_height, (1280, 720))
        self.assertEqual(train_assets[2].quality_width_height, (1280, 720))
        self.assertEqual(train_assets[2].dis_yuv_type, 'yuv420p10le')

    def test_read_image_dataset_notyuv(self):
        dataset_path = VmafConfig.test_resource_path('test_image_dataset_notyuv.py')
        dataset = import_python_file(dataset_path)
        assets = read_dataset(dataset)

        self.assertEqual(len(assets), 4)
        self.assertTrue(assets[0].ref_width_height is None)
        self.assertTrue(assets[0].dis_width_height is None)
        self.assertEqual(assets[0].workfile_yuv_type, assets[0].DEFAULT_YUV_TYPE)
        self.assertEqual(assets[0].quality_width_height, (1920, 1080))

    def test_read_image_dataset_notyuv_workfile_yuv_type(self):
        dataset_path = VmafConfig.test_resource_path('test_image_dataset_notyuv_workfile_yuv_type.py')
        dataset = import_python_file(dataset_path)
        assets = read_dataset(dataset)

        self.assertEqual(len(assets), 4)
        self.assertTrue(assets[0].ref_width_height is None)
        self.assertTrue(assets[0].dis_width_height is None)
        self.assertEqual(assets[0].quality_width_height, (1920, 1080))
        self.assertEqual(assets[0].workfile_yuv_type, 'yuv444p')

    def test_read_dataset_basic(self):
        dataset_path = VmafConfig.test_resource_path('test_dataset.py')
        dataset = import_python_file(dataset_path)
        assets = read_dataset(dataset)

        self.assertEqual(len(assets), 4)
        self.assertTrue('groundtruth' in assets[0].asset_dict.keys())
        self.assertTrue('os' not in assets[0].asset_dict.keys())
        self.assertEqual(assets[0].quality_width_height, (1920, 1080))
        self.assertEqual(assets[0].resampling_type, 'bicubic')
        self.assertEqual(assets[0].ref_yuv_type, 'yuv420p')
        self.assertEqual(assets[0].dis_yuv_type, 'yuv420p')
        self.assertEqual(assets[1].quality_width_height, (1920, 1080))
        self.assertEqual(assets[1].resampling_type, 'bicubic')

    def test_read_dataset_mixed(self):
        dataset_path = VmafConfig.test_resource_path('test_dataset_mixed.py')
        dataset = import_python_file(dataset_path)
        assets = read_dataset(dataset)

        self.assertEqual(len(assets), 4)

        self.assertEqual(assets[0].resampling_type, 'bicubic')
        self.assertEqual(assets[0].ref_yuv_type, 'yuv420p')
        self.assertEqual(assets[0].dis_yuv_type, 'yuv420p')
        self.assertEqual(assets[0].ref_width_height, (1920, 1080))
        self.assertEqual(assets[0].dis_width_height, (1920, 1080))

        self.assertEqual(assets[1].resampling_type, 'bicubic')
        self.assertEqual(assets[1].ref_yuv_type, 'yuv420p')
        self.assertEqual(assets[1].dis_yuv_type, 'notyuv')
        self.assertEqual(assets[1].ref_width_height, (1920, 1080))
        self.assertEqual(assets[1].dis_width_height, None)

        self.assertEqual(assets[2].resampling_type, 'bicubic')
        self.assertEqual(assets[2].ref_yuv_type, 'yuv420p')
        self.assertEqual(assets[2].dis_yuv_type, 'yuv420p')
        self.assertEqual(assets[2].ref_width_height, (720, 480))
        self.assertEqual(assets[2].dis_width_height, (720, 480))

        self.assertEqual(assets[3].resampling_type, 'bicubic')
        self.assertEqual(assets[3].ref_yuv_type, 'yuv420p')
        self.assertEqual(assets[3].dis_yuv_type, 'notyuv')
        self.assertEqual(assets[3].ref_width_height, (720, 480))
        self.assertEqual(assets[3].dis_width_height, None)

    def test_read_dataset_start_end_frame_crop(self):
        dataset_path = VmafConfig.test_resource_path('test_read_dataset_dataset.py')
        dataset = import_python_file(dataset_path)
        assets = read_dataset(dataset)

        assets[0].asset_dict['ref_crop_cmd'] = '1280:1920:0:0'

        self.assertEqual(len(assets), 3)

        self.assertEqual(assets[0].resampling_type, 'bicubic')
        self.assertEqual(assets[0].ref_yuv_type, 'notyuv')
        self.assertEqual(assets[0].dis_yuv_type, 'notyuv')
        self.assertEqual(assets[0].ref_width_height, None)
        self.assertEqual(assets[0].dis_width_height, None)
        self.assertEqual(assets[0].ref_start_end_frame, (200, 210))
        self.assertEqual(assets[0].dis_start_end_frame, (200, 210))
        self.assertEqual(assets[0].ref_crop_cmd, '1280:1920:0:0')
        self.assertEqual(assets[0].dis_crop_cmd, '1280:1920:0:0')
        self.assertEqual(assets[1].ref_start_end_frame, (200, 210))
        self.assertEqual(assets[1].dis_start_end_frame, (200, 210))

    def test_read_dataset_refdif_start_end_frame(self):
        dataset_path = VmafConfig.test_resource_path('test_read_dataset_dataset3.py')
        dataset = import_python_file(dataset_path)
        assets = read_dataset(dataset)

        self.assertEqual(len(assets), 3)

        self.assertEqual(assets[0].ref_start_end_frame, (250, 250))
        self.assertEqual(assets[0].dis_start_end_frame, (250, 250))
        self.assertEqual(assets[1].ref_start_end_frame, (250, 250))
        self.assertEqual(assets[1].dis_start_end_frame, (200, 210))
        self.assertEqual(assets[2].ref_start_end_frame, (250, 250))
        self.assertEqual(assets[2].dis_start_end_frame, (250, 251))

    def test_read_dataset_fps_duration_sec(self):
        dataset_path = VmafConfig.test_resource_path('test_read_dataset_dataset2.py')
        dataset = import_python_file(dataset_path)
        assets = read_dataset(dataset)

        assets[0].asset_dict['ref_crop_cmd'] = '1280:1920:0:0'

        self.assertEqual(len(assets), 3)

        self.assertEqual(assets[0].resampling_type, 'bicubic')
        self.assertEqual(assets[0].ref_yuv_type, 'notyuv')
        self.assertEqual(assets[0].dis_yuv_type, 'notyuv')
        self.assertEqual(assets[0].ref_width_height, None)
        self.assertEqual(assets[0].dis_width_height, None)
        self.assertEqual(assets[0].ref_start_end_frame, (0, 9))
        self.assertEqual(assets[0].dis_start_end_frame, (0, 9))
        self.assertEqual(assets[0].ref_crop_cmd, '1280:1920:0:0')
        self.assertEqual(assets[0].dis_crop_cmd, '1280:1920:0:0')
        self.assertEqual(assets[1].ref_start_end_frame, (100, 110))
        self.assertEqual(assets[1].dis_start_end_frame, (100, 110))


class TestTrainOnDatasetJsonFormat(unittest.TestCase):

    def setUp(self):
        self.output_model_filepath = VmafConfig.workspace_path("model", "test_output_model.json")

    def tearDown(self):
        if os.path.exists(self.output_model_filepath):
            os.remove(self.output_model_filepath)

    def test_train_test_on_dataset_with_dis1st_thr(self):
        from vmaf.routine import train_test_vmaf_on_dataset
        train_dataset = import_python_file(
            VmafConfig.test_resource_path('dataset_sample.py'))
        model_param = import_python_file(
            VmafConfig.test_resource_path('model_param_sample.py'))
        feature_param = import_python_file(
            VmafConfig.test_resource_path('feature_param_sample.py'))

        train_fassembler, train_assets, train_stats, test_fassembler, test_assets, test_stats, _ = train_test_vmaf_on_dataset(
            train_dataset=train_dataset,
            test_dataset=train_dataset,
            feature_param=feature_param,
            model_param=model_param,
            train_ax=None,
            test_ax=None,
            result_store=None,
            parallelize=False,
            logger=None,
            fifo_mode=False,
            output_model_filepath=self.output_model_filepath,
        )

        self.train_fassembler = train_fassembler
        self.assertTrue(os.path.exists(self.output_model_filepath))
        self.assertAlmostEqual(train_stats['ys_label_pred'][0], 90.753010402770798, places=3)
        self.assertAlmostEqual(test_stats['ys_label_pred'][0], 90.753010402770798, places=3)


class TestTrainOnDataset(unittest.TestCase):

    def setUp(self):
        self.output_model_filepath = VmafConfig.workspace_path("model", "test_output_model.pkl")

    def tearDown(self):
        if os.path.exists(self.output_model_filepath):
            os.remove(self.output_model_filepath)

    def test_train_test_on_dataset_with_dis1st_thr(self):
        from vmaf.routine import train_test_vmaf_on_dataset
        train_dataset = import_python_file(
            VmafConfig.test_resource_path('dataset_sample.py'))
        model_param = import_python_file(
            VmafConfig.test_resource_path('model_param_sample.py'))
        feature_param = import_python_file(
            VmafConfig.test_resource_path('feature_param_sample.py'))

        train_fassembler, train_assets, train_stats, test_fassembler, test_assets, test_stats, _ = train_test_vmaf_on_dataset(
            train_dataset=train_dataset,
            test_dataset=train_dataset,
            feature_param=feature_param,
            model_param=model_param,
            train_ax=None,
            test_ax=None,
            result_store=None,
            parallelize=False,
            logger=None,
            fifo_mode=False,
            output_model_filepath=self.output_model_filepath,
        )

        self.train_fassembler = train_fassembler
        self.assertTrue(os.path.exists(self.output_model_filepath))
        self.assertAlmostEqual(train_stats['ys_label_pred'][0], 90.753010402770798, places=3)
        self.assertAlmostEqual(test_stats['ys_label_pred'][0], 90.753010402770798, places=3)

        runner = VmafQualityRunner(
            train_assets,
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': self.output_model_filepath}
        )
        runner.run(parallelize=False)
        results = runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 89.55494473011981, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 61.025217541256076, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 90.75301241304798, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 89.27013895870179, places=4)

    def test_train_test_on_raw_dataset_with_dis1st_thr(self):
        from vmaf.routine import train_test_vmaf_on_dataset
        train_dataset = import_python_file(
            VmafConfig.test_resource_path('raw_dataset_sample.py'))
        model_param = import_python_file(
            VmafConfig.test_resource_path('model_param_sample.py'))
        feature_param = import_python_file(
            VmafConfig.test_resource_path('feature_param_sample.py'))

        train_fassembler, train_assets, train_stats, test_fassembler, test_assets, test_stats, _ = train_test_vmaf_on_dataset(
            train_dataset=train_dataset,
            test_dataset=train_dataset,
            feature_param=feature_param,
            model_param=model_param,
            train_ax=None,
            test_ax=None,
            result_store=None,
            parallelize=False,
            logger=None,
            fifo_mode=False,
            output_model_filepath=self.output_model_filepath
        )

        self.train_fassembler = train_fassembler
        self.assertTrue(os.path.exists(self.output_model_filepath))
        self.assertAlmostEqual(train_stats['ys_label_pred'][0], 93.565459224020742, places=3)
        self.assertAlmostEqual(test_stats['ys_label_pred'][0], 93.565459224020742, places=3)

    def test_test_on_dataset(self):
        from vmaf.routine import run_test_on_dataset
        test_dataset = import_python_file(
            VmafConfig.test_resource_path('dataset_sample.py'))
        test_assets, results = run_test_on_dataset(test_dataset, VmafQualityRunner, None,
                                                   None, VmafConfig.model_path("vmaf_float_v0.6.1.json"),
                                                   parallelize=False,
                                                   aggregate_method=None)

        self.assertAlmostEqual(results[0]['VMAF_score'], 99.142659046424384, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 35.066157497128764, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 97.428042675471147, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 97.427927701008869, places=4)
        self.assertAlmostEqual(test_assets[0].groundtruth, 100, places=4)
        self.assertAlmostEqual(test_assets[1].groundtruth, 50, places=4)
        self.assertAlmostEqual(test_assets[2].groundtruth, 100, places=4)
        self.assertAlmostEqual(test_assets[3].groundtruth, 80, places=4)

    @unittest.skip("Inconsistent numerical values.")
    def test_compare_two_quality_runners_on_dataset(self):
        test_dataset = import_python_file(VmafConfig.test_resource_path('dataset_sample.py'))
        result = compare_two_quality_runners_on_dataset(
            test_dataset, VmafQualityRunner, PsnrQualityRunner,
            result_store=None,
            num_resample=10,
            seed_resample=0,
            subj_model_class=SubjectiveModel.find_subclass('MOS'),
            parallelize=False, fifo_mode=False)
        self.assertAlmostEqual(np.nanmean(list(zip(*result['plcc']))[0]), 0.8655928449687122, places=4)
        self.assertAlmostEqual(np.nanmean(list(zip(*result['plcc']))[1]), 0.9875440797696373, places=4)
        self.assertAlmostEqual(np.nanmean(list(zip(*result['srocc']))[0]), 0.8642507701111302, places=4)
        self.assertAlmostEqual(np.nanmean(list(zip(*result['srocc']))[1]), 1.0, places=4)

        self.assertTrue(np.isnan(result['plcc_ci95_first'][0]))
        self.assertTrue(np.isnan(result['plcc_ci95_first'][1]))
        self.assertTrue(np.isnan(result['plcc_ci95_second'][0]))
        self.assertTrue(np.isnan(result['plcc_ci95_second'][1]))
        self.assertTrue(np.isnan(result['plcc_ci95_diff'][0]))
        self.assertTrue(np.isnan(result['plcc_ci95_diff'][1]))

        self.assertTrue(np.isnan(result['srocc_ci95_first'][0]))
        self.assertTrue(np.isnan(result['srocc_ci95_first'][1]))
        self.assertTrue(np.isnan(result['srocc_ci95_second'][0]))
        self.assertTrue(np.isnan(result['srocc_ci95_second'][1]))
        self.assertTrue(np.isnan(result['srocc_ci95_diff'][0]))
        self.assertTrue(np.isnan(result['srocc_ci95_diff'][1]))

    def test_test_on_dataset_plot_per_content(self):
        from vmaf.routine import run_test_on_dataset
        test_dataset = import_python_file(
            VmafConfig.test_resource_path('dataset_sample.py'))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=[20, 20])
        run_test_on_dataset(test_dataset, VmafQualityRunner, ax,
                            None, VmafConfig.model_path("vmaf_float_v0.6.1.json"),
                            parallelize=False,
                            fifo_mode=False,
                            aggregate_method=None,
                            point_label='asset_id',
                            do_plot=['aggregate',  # plots all contents in one figure
                                     'per_content',  # plots a separate figure per content
                                     'groundtruth_predicted_in_parallel',  # plots of groundtruth and predicted in parallel
                                     ],
                            plot_linear_fit=True  # adds linear fit line to each plot
                            )

        output_dir = VmafConfig.workspace_path("output", "test_output")
        DisplayConfig.show(write_to_dir=output_dir)
        self.assertEqual(len(glob.glob(os.path.join(output_dir, '*.png'))), 4)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    def test_test_on_dataset_bootstrap_quality_runner(self):
        from vmaf.routine import run_test_on_dataset
        test_dataset = import_python_file(
            VmafConfig.test_resource_path('dataset_sample.py'))
        test_assets, results = run_test_on_dataset(test_dataset, BootstrapVmafQualityRunner, None,
                                                   None, VmafConfig.model_path("vmaf_float_b_v0.6.3.json"),
                                                   parallelize=False,
                                                   aggregate_method=None)

        expecteds = [98.7927560599655, 100.0, 100.0, 98.82959541116277, 99.80711961053976, 98.91713244333198, 100.0,
                     99.33233498293374, 98.99337537979711, 99.62668672314118, 99.00885507364698, 100.0, 97.29492843378944,
                     100.0, 99.02101642563275, 94.50521964145268, 95.63007904351339, 98.57370486684022, 100.0,
                     99.36754906446309]

        actuals = results[0]['BOOTSTRAP_VMAF_all_models_score']

        assert len(actuals) == len(expecteds), "Expected and actual bootstrap prediction lists do not match in length."

        for actual, expected in zip(actuals, expecteds):
            self.assertAlmostEqual(actual, expected, places=4)
        self.assertAlmostEqual(results[0]['BOOTSTRAP_VMAF_score'], 99.32876664539778, places=4)

    def test_test_on_dataset_split_test_indices_for_perf_ci(self):
        from vmaf.routine import run_test_on_dataset
        test_dataset = import_python_file(VmafConfig.test_resource_path('dataset_sample.py'))
        test_assets, results = run_test_on_dataset(test_dataset, VmafQualityRunner, None,
                                                   None, VmafConfig.model_path("vmaf_float_v0.6.1.json"), parallelize=False,
                                                   aggregate_method=None,
                                                   split_test_indices_for_perf_ci=True,
                                                   n_splits_test_indices=10)

        self.assertAlmostEqual(results[0]['VMAF_score'], 99.142659046424384, places=4)

    def test_test_on_dataset_raw(self):
        from vmaf.routine import run_test_on_dataset
        test_dataset = import_python_file(
            VmafConfig.test_resource_path('raw_dataset_sample.py'))
        test_assets, results = run_test_on_dataset(test_dataset, VmafQualityRunner, None,
                                                   None, VmafConfig.model_path("vmaf_float_v0.6.1.json"),
                                                   parallelize=False,
                                                   aggregate_method=None)

        self.assertAlmostEqual(results[0]['VMAF_score'], 99.142659046424384, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 35.066157497128764, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 97.428042675471147, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 97.427927701008869, places=4)
        self.assertAlmostEqual(test_assets[0].groundtruth, 100, places=4)
        self.assertAlmostEqual(test_assets[1].groundtruth, 50, places=4)
        self.assertAlmostEqual(test_assets[2].groundtruth, 100, places=4)
        self.assertAlmostEqual(test_assets[3].groundtruth, 90, places=4)
        self.assertAlmostEqual(test_assets[0].groundtruth_std, 0.0, places=4)
        self.assertAlmostEqual(test_assets[1].groundtruth_std, 3.5355339059327373, places=4)
        self.assertAlmostEqual(test_assets[2].groundtruth_std, 0.0, places=4)
        self.assertAlmostEqual(test_assets[3].groundtruth_std, 3.5355339059327373, places=4)

    def test_test_on_dataset_mle(self):
        from vmaf.routine import run_test_on_dataset
        test_dataset = import_python_file(
            VmafConfig.test_resource_path('raw_dataset_sample.py'))
        test_assets, results = run_test_on_dataset(test_dataset, VmafQualityRunner, None,
                                                   None, VmafConfig.model_path("vmaf_float_v0.6.1.json"),
                                                   parallelize=False,
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
        self.assertAlmostEqual(test_assets[0].groundtruth_std, 0.0, places=4)
        self.assertAlmostEqual(test_assets[1].groundtruth_std, 3.5355339059327373, places=4)
        self.assertAlmostEqual(test_assets[2].groundtruth_std, 0.0, places=4)
        self.assertAlmostEqual(test_assets[3].groundtruth_std, 3.5355339059327373, places=4)

    def test_train_test_on_dataset_with_dis1st_thr_with_feature_optional_dict(self):
        from vmaf.routine import train_test_vmaf_on_dataset
        train_dataset = import_python_file(
            VmafConfig.test_resource_path('dataset_sample.py'))
        model_param = import_python_file(
            VmafConfig.test_resource_path('model_param_sample.py'))
        feature_param = import_python_file(
            VmafConfig.test_resource_path('feature_param_sample_with_optional_dict.py'))

        with self.assertRaises(AssertionError):
            # adm_ref_display_height 108000 exceeds the maximum allowed
            train_fassembler, train_assets, train_stats, test_fassembler, test_assets, test_stats, _ = train_test_vmaf_on_dataset(
                train_dataset=train_dataset,
                test_dataset=train_dataset,
                feature_param=feature_param,
                model_param=model_param,
                train_ax=None,
                test_ax=None,
                result_store=None,
                parallelize=False,
                logger=None,
                fifo_mode=False,
                output_model_filepath=self.output_model_filepath,
            )

    def test_train_test_on_dataset_with_dis1st_thr_with_feature_optional_dict_good(self):
        from vmaf.routine import train_test_vmaf_on_dataset
        train_dataset = import_python_file(
            VmafConfig.test_resource_path('dataset_sample.py'))
        model_param = import_python_file(
            VmafConfig.test_resource_path('model_param_sample.py'))
        feature_param = import_python_file(
            VmafConfig.test_resource_path('feature_param_sample_with_optional_dict_good.py'))

        train_fassembler, train_assets, train_stats, test_fassembler, test_assets, test_stats, _ = train_test_vmaf_on_dataset(
            train_dataset=train_dataset,
            test_dataset=train_dataset,
            feature_param=feature_param,
            model_param=model_param,
            train_ax=None,
            test_ax=None,
            result_store=None,
            parallelize=False,
            logger=None,
            fifo_mode=False,
            output_model_filepath=self.output_model_filepath,
        )

        self.train_fassembler = train_fassembler
        self.assertTrue(os.path.exists(self.output_model_filepath))
        self.assertAlmostEqual(train_stats['ys_label_pred'][0], 90.753010402770798, places=3)
        self.assertAlmostEqual(test_stats['ys_label_pred'][0], 90.753010402770798, places=3)

        runner = VmafQualityRunner(
            train_assets,
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': self.output_model_filepath}
        )
        runner.run(parallelize=False)
        results = runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 89.55494473011981, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 61.01289549048653, places=4)
        self.assertAlmostEqual(results[2]['VMAF_score'], 90.75301241304798, places=4)
        self.assertAlmostEqual(results[3]['VMAF_score'], 89.27013895870179, places=4)


class TestGenerateDatasetFromRaw(unittest.TestCase):

    def setUp(self):
        self.raw_dataset_filepath = VmafConfig.resource_path("dataset", "NFLX_dataset_public_raw.py")
        self.derived_dataset_path = VmafConfig.workdir_path("test_derived_dataset.py")
        self.derived_dataset_path_pyc = VmafConfig.workdir_path("test_derived_dataset.pyc")

    def tearDown(self):
        if os.path.exists(self.derived_dataset_path):
            os.remove(self.derived_dataset_path)
        if os.path.exists(self.derived_dataset_path_pyc):
            os.remove(self.derived_dataset_path_pyc)

    def test_generate_dataset_from_raw_default(self):  # DMOS
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
    unittest.main(verbosity=2)
