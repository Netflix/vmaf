import numpy as np

import unittest
import config
from core.asset import Asset
from core.local_explainer import VmafQualityRunnerWithLocalExplainer, \
    LocalExplainer

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class LocalExplainerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
            pass

    def setUp(self):
        pass

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
            optional_dict2={'explainer': LocalExplainer(neighbor_samples=100,
                                                        neighbor_std=1.0
                                                        )}
        )

        np.random.seed(0)

        self.runner.run()
        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 65.4488588759, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.2259317881, places=4)

        weights = np.mean(results[0]['VMAF_scores_exp']['feature_weights'], axis=0)
        self.assertAlmostEqual(weights[0], 0.75441663, places=4)
        self.assertAlmostEqual(weights[1], 0.06816105, places=4)
        self.assertAlmostEqual(weights[2], -0.10934421, places=4)
        self.assertAlmostEqual(weights[3], 0.22051127, places=4)
        self.assertAlmostEqual(weights[4], 0.12517884, places=4)
        self.assertAlmostEqual(weights[5], 0.04639162, places=4)

        weights = np.mean(results[1]['VMAF_scores_exp']['feature_weights'], axis=0)
        self.assertAlmostEqual(weights[0], 0.77096087, places=4)
        self.assertAlmostEqual(weights[1], 0.01491754, places=4)
        self.assertAlmostEqual(weights[2], -0.08025557, places=4)
        self.assertAlmostEqual(weights[3], 0.2511188, places=4)
        self.assertAlmostEqual(weights[4], 0.14953561, places=4)
        self.assertAlmostEqual(weights[5], 0.07960753, places=4)
