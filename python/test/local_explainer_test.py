import os
import unittest

import numpy as np

import config
from core.asset import Asset
from core.executor import run_executors_in_parallel
from core.local_explainer import LocalExplainer
from core.quality_runner_extra import VmafQualityRunnerWithLocalExplainer
from core.noref_feature_extractor import MomentNorefFeatureExtractor
from core.raw_extractor import DisYUVRawVideoExtractor
from core.train_test_model import SklearnRandomForestTrainTestModel, \
    MomentRandomForestTrainTestModel
from routine import read_dataset
from tools.misc import import_python_file

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class LocalExplainerTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def test_explain_train_test_model(self):

        model_class = SklearnRandomForestTrainTestModel

        train_dataset_path = config.ROOT + '/python/test/resource/' \
                                           'test_image_dataset_diffdim.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)
        _, self.features = run_executors_in_parallel(
            MomentNorefFeatureExtractor,
            train_assets,
            fifo_mode=True,
            delete_workdir=True,
            parallelize=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=None,
        )

        xys = model_class.get_xys_from_results(self.features[:7])
        model = model_class({'norm_type':'normalize', 'random_state':0}, None)
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
        print 'test on running VMAF runner with local explainer...'
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324})

        self.runner = VmafQualityRunnerWithLocalExplainer(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict2={'explainer': LocalExplainer(neighbor_samples=100)}
        )

        np.random.seed(0)

        self.runner.run()
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 65.4488588759, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.2259317881, places=4)

        expected_feature_names = ['VMAF_feature_adm2_score',
                                  'VMAF_feature_motion_score',
                                  'VMAF_feature_vif_scale0_score',
                                  'VMAF_feature_vif_scale1_score',
                                  'VMAF_feature_vif_scale2_score',
                                  'VMAF_feature_vif_scale3_score']

        weights = np.mean(results[0]['VMAF_scores_exps']['feature_weights'], axis=0)
        self.assertAlmostEqual(weights[0], 0.75441663, places=4)
        self.assertAlmostEqual(weights[1], 0.06816105, places=4)
        self.assertAlmostEqual(weights[2], -0.10934421, places=4)
        self.assertAlmostEqual(weights[3], 0.22051127, places=4)
        self.assertAlmostEqual(weights[4], 0.12517884, places=4)
        self.assertAlmostEqual(weights[5], 0.04639162, places=4)

        self.assertEqual(results[0]['VMAF_scores_exps']['feature_names'],
                         expected_feature_names)

        weights = np.mean(results[1]['VMAF_scores_exps']['feature_weights'], axis=0)
        self.assertAlmostEqual(weights[0], 0.77096087, places=4)
        self.assertAlmostEqual(weights[1], 0.01491754, places=4)
        self.assertAlmostEqual(weights[2], -0.08025557, places=4)
        self.assertAlmostEqual(weights[3], 0.2511188, places=4)
        self.assertAlmostEqual(weights[4], 0.14953561, places=4)
        self.assertAlmostEqual(weights[5], 0.07960753, places=4)

        self.assertEqual(results[1]['VMAF_scores_exps']['feature_names'],
                         expected_feature_names)

        # self.runner.show_local_explanations(results, indexs=[2, 3])
        # import matplotlib.pyplot as plt
        # plt.show()


class LocalExplainerMomentRandomForestTest(unittest.TestCase):

    def setUp(self):
        train_dataset_path = config.ROOT + '/python/test/resource/test_image_dataset_diffdim.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.h5py_filepath = config.ROOT + '/workspace/workdir/test.hdf5'
        self.h5py_file = DisYUVRawVideoExtractor.open_h5py_file(self.h5py_filepath)
        optional_dict2 = {'h5py_file': self.h5py_file}

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

        model = model_class({'norm_type':'normalize', 'random_state':0})
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


if __name__ == '__main__':
    unittest.main()
