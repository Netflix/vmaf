import os
import unittest

import numpy as np

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.local_explainer import LocalExplainer
from vmaf.core.quality_runner_extra import VmafQualityRunnerWithLocalExplainer
from vmaf.core.noref_feature_extractor import MomentNorefFeatureExtractor
from vmaf.core.raw_extractor import DisYUVRawVideoExtractor
from vmaf.core.result_store import FileSystemResultStore
from vmaf.core.train_test_model import SklearnRandomForestTrainTestModel, \
    MomentRandomForestTrainTestModel
from vmaf.routine import read_dataset
from vmaf.tools.misc import import_python_file

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class LocalExplainerTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def test_explain_train_test_model(self):

        model_class = SklearnRandomForestTrainTestModel

        train_dataset_path = VmafConfig.test_resource_path('test_image_dataset_diffdim2.py')
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        fextractor = MomentNorefFeatureExtractor(
            train_assets,
            None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=None,
        )
        fextractor.run(parallelize=True)
        self.features = fextractor.results

        xys = model_class.get_xys_from_results(self.features[:7])
        model = model_class({'norm_type': 'normalize',
                             'n_estimators': 10,
                             'random_state': 0}, None)
        model.train(xys)

        np.random.seed(0)

        xs = model_class.get_xs_from_results(self.features[7:])
        explainer = LocalExplainer(neighbor_samples=1000)
        exps = explainer.explain(model, xs)

        self.assertAlmostEqual(exps['feature_weights'][0, 0], -0.12416, places=4)
        self.assertAlmostEqual(exps['feature_weights'][1, 0], 0.00076, places=4)
        self.assertAlmostEqual(exps['feature_weights'][0, 1], -0.20931, places=4)
        self.assertAlmostEqual(exps['feature_weights'][1, 1], -0.01245, places=4)
        self.assertAlmostEqual(exps['feature_weights'][0, 2], 0.02322, places=4)
        self.assertAlmostEqual(exps['feature_weights'][1, 2], 0.03673, places=4)

        self.assertAlmostEqual(exps['features'][0, 0], 107.73501, places=4)
        self.assertAlmostEqual(exps['features'][1, 0], 35.81638, places=4)
        self.assertAlmostEqual(exps['features'][0, 1], 13691.23881, places=4)
        self.assertAlmostEqual(exps['features'][1, 1], 1611.56764, places=4)
        self.assertAlmostEqual(exps['features'][0, 2], 2084.40542, places=4)
        self.assertAlmostEqual(exps['features'][1, 2], 328.75389, places=4)

        self.assertAlmostEqual(exps['features_normalized'][0, 0], -0.65527, places=4)
        self.assertAlmostEqual(exps['features_normalized'][1, 0], -3.74922, places=4)
        self.assertAlmostEqual(exps['features_normalized'][0, 1], -0.68872, places=4)
        self.assertAlmostEqual(exps['features_normalized'][1, 1], -2.79586, places=4)
        self.assertAlmostEqual(exps['features_normalized'][0, 2], 0.08524, places=4)
        self.assertAlmostEqual(exps['features_normalized'][1, 2], -1.32625, places=4)

        self.assertEqual(exps['feature_names'],
                         ['Moment_noref_feature_1st_score',
                          'Moment_noref_feature_2nd_score',
                          'Moment_noref_feature_var_score']
                         )

    def test_explain_vmaf_results(self):
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324})

        self.runner = VmafQualityRunnerWithLocalExplainer(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path("vmaf_float_v0.6.1.json")},
            optional_dict2={'explainer': LocalExplainer(neighbor_samples=100)}
        )

        np.random.seed(0)

        self.runner.run()
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_LE_score'], 76.68425574067017, places=4)
        self.assertAlmostEqual(results[1]['VMAF_LE_score'], 99.946416604585025, places=4)

        expected_feature_names = ['VMAF_feature_adm2_score',
                                  'VMAF_feature_motion2_score',
                                  'VMAF_feature_vif_scale0_score',
                                  'VMAF_feature_vif_scale1_score',
                                  'VMAF_feature_vif_scale2_score',
                                  'VMAF_feature_vif_scale3_score']

        weights = np.mean(results[0]['VMAF_LE_scores_exps']['feature_weights'], axis=0)
        self.assertAlmostEqual(weights[0], 0.66021689480916868, places=4)
        self.assertAlmostEqual(weights[1], 0.14691682562211777, places=4)
        self.assertAlmostEqual(weights[2], -0.023682744847036086, places=4)
        self.assertAlmostEqual(weights[3], -0.029779341850172818, places=4)
        self.assertAlmostEqual(weights[4], 0.19149485210137338, places=4)
        self.assertAlmostEqual(weights[5], 0.31890978778344126, places=4)

        self.assertEqual(results[0]['VMAF_LE_scores_exps']['feature_names'],
                         expected_feature_names)

        weights = np.mean(results[1]['VMAF_LE_scores_exps']['feature_weights'], axis=0)
        self.assertAlmostEqual(weights[0], 0.69597961598838509, places=4)
        self.assertAlmostEqual(weights[1], 0.18256016705513464, places=4)
        self.assertAlmostEqual(weights[2], 0.0090048099912423147, places=4)
        self.assertAlmostEqual(weights[3], 0.028671810808880094, places=4)
        self.assertAlmostEqual(weights[4], 0.21935602577417926, places=4)
        self.assertAlmostEqual(weights[5], 0.34190431429767715, places=4)

        self.assertEqual(results[1]['VMAF_LE_scores_exps']['feature_names'],
                         expected_feature_names)

        # self.runner.show_local_explanations(results, indexs=[2, 3])
        # import matplotlib.pyplot as plt
        # DisplayConfig.show()


class LocalExplainerMomentRandomForestTest(unittest.TestCase):

    def setUp(self):
        train_dataset_path = VmafConfig.test_resource_path("test_image_dataset_diffdim2.py")
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.h5py_filepath = VmafConfig.workdir_path('test.hdf5')
        self.h5py_file = DisYUVRawVideoExtractor.open_h5py_file(self.h5py_filepath)
        optional_dict2 = {'h5py_file': self.h5py_file}

        fextractor = DisYUVRawVideoExtractor(
            train_assets,
            None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=optional_dict2,
        )
        fextractor.run(parallelize=False)  # CAN ONLY USE SERIAL MODE FOR DisYRawVideoExtractor
        self.features = fextractor.results

    def tearDown(self):
        if hasattr(self, 'h5py_file'):
            DisYUVRawVideoExtractor.close_h5py_file(self.h5py_file)
        if os.path.exists(self.h5py_filepath):
            os.remove(self.h5py_filepath)

    def test_explain_train_test_model(self):

        model_class = MomentRandomForestTrainTestModel

        xys = model_class.get_xys_from_results(self.features[:7])
        del xys['dis_u']
        del xys['dis_v']

        model = model_class({'norm_type': 'normalize',
                             'n_estimators': 10,
                             'random_state': 0})
        model.train(xys)

        np.random.seed(0)

        xs = model_class.get_xs_from_results(self.features[7:])
        del xs['dis_u']
        del xs['dis_v']

        explainer = LocalExplainer(neighbor_samples=1000)
        exps = explainer.explain(model, xs)

        self.assertAlmostEqual(exps['feature_weights'][0, 0], -0.12416, places=4)
        self.assertAlmostEqual(exps['feature_weights'][1, 0], 0.00076, places=4)
        self.assertAlmostEqual(exps['feature_weights'][0, 1], -0.20931, places=4)
        self.assertAlmostEqual(exps['feature_weights'][1, 1], -0.01245, places=4)
        self.assertAlmostEqual(exps['feature_weights'][0, 2], 0.02322, places=4)
        self.assertAlmostEqual(exps['feature_weights'][1, 2], 0.03673, places=4)

        self.assertAlmostEqual(exps['features'][0, 0], 107.73501, places=4)
        self.assertAlmostEqual(exps['features'][1, 0], 35.81638, places=4)
        self.assertAlmostEqual(exps['features'][0, 1], 13691.23881, places=4)
        self.assertAlmostEqual(exps['features'][1, 1], 1611.56764, places=4)
        self.assertAlmostEqual(exps['features'][0, 2], 2084.40542, places=4)
        self.assertAlmostEqual(exps['features'][1, 2], 328.75389, places=4)

        self.assertAlmostEqual(exps['features_normalized'][0, 0], -0.65527, places=4)
        self.assertAlmostEqual(exps['features_normalized'][1, 0], -3.74922, places=4)
        self.assertAlmostEqual(exps['features_normalized'][0, 1], -0.68872, places=4)
        self.assertAlmostEqual(exps['features_normalized'][1, 1], -2.79586, places=4)
        self.assertAlmostEqual(exps['features_normalized'][0, 2], 0.08524, places=4)
        self.assertAlmostEqual(exps['features_normalized'][1, 2], -1.32625, places=4)

        self.assertEqual(exps['feature_names'], ['dis_y'])
        # TODO: fix feature name to 'Moment_noref_feature_1st_score', ...


class QualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        self.result_store = FileSystemResultStore()

    def test_run_vmaf_runner_local_explainer_with_bootstrap_model(self):
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324})

        self.runner = VmafQualityRunnerWithLocalExplainer(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={
                'model_filepath': VmafConfig.test_resource_path('model', 'vmafplus_v0.5.2_test.json'),
            },
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_LE_score'], 75.42800743529182, places=4)
        self.assertAlmostEqual(results[1]['VMAF_LE_score'], 99.95804893252175, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
