from __future__ import absolute_import

import os
import sys
import unittest

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset, NorefAsset
from vmaf.core.quality_runner import VmafLegacyQualityRunner, VmafQualityRunner, \
    PsnrQualityRunner, MsSsimQualityRunner, \
    SsimQualityRunner, Adm2QualityRunner, VmafPhoneQualityRunner, VifQualityRunner, \
    Vif2QualityRunner, BootstrapVmafQualityRunner, BaggingVmafQualityRunner, NiqeQualityRunner, \
    EnsembleVmafQualityRunner, VmafnegQualityRunner
from vmaf.core.result_store import FileSystemResultStore
from vmaf.tools.misc import MyTestCase
from vmaf.tools.stats import ListStats

from test.testutil import set_default_576_324_videos_for_testing, set_default_flat_1920_1080_videos_for_testing, \
    set_default_576_324_10bit_videos_for_testing, set_default_576_324_12bit_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class QualityRunnerVersionTest(unittest.TestCase):

    def test_vmaf_quality_runner_version(self):
        self.assertEqual(VmafQualityRunner.VERSION, 'F0.2.7int-0.6.1')
        self.assertEqual(VmafQualityRunner.ALGO_VERSION, 4)


class QualityRunnerResultStoreTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        self.runner1.remove_results()
        self.runner2.remove_results()
        super().tearDown()

    def test_quality_runner_with_different_models(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner1 = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_rdh540.json')}
        )
        self.runner1.run(parallelize=True)
        results1 = self.runner1.results

        self.runner2 = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=self.result_store,
            optional_dict={'model_filepath': VmafConfig.test_resource_path('model', 'vmaf_float_v0.6.1_nvd6.json')}
        )
        self.runner2.run(parallelize=True)
        results2 = self.runner2.results

        self.assertAlmostEqual(results1[0]['VMAF_score'], 73.28968543912883, places=4)
        self.assertAlmostEqual(results1[1]['VMAF_score'], 99.946416604585025, places=4)

        self.assertAlmostEqual(results2[0]['VMAF_score'], 80.61670115719328, places=4)
        self.assertAlmostEqual(results2[1]['VMAF_score'], 99.946416604585025, places=4)


class QualityRunnerSaveWorkfilesTest(MyTestCase):

    def setUp(self):
        super().setUp()
        self.result_store = FileSystemResultStore()

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
        super().tearDown()

    def test_run_vmaf_runner_flat_save_workfiles(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = VmafQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=self.result_store,
            save_workfiles=True,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAF_score'], 76.66890511746402, places=4)
        self.assertAlmostEqual(results[1]['VMAF_score'], 99.94642662500576, places=4)

    def test_run_psnr_runner_flat_save_workfiles(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.runner = PsnrQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=self.result_store,
            save_workfiles=True,
        )
        self.runner.run(parallelize=False)

        results = self.runner.results

        self.assertAlmostEqual(results[0]['PSNR_score'], 30.755063979166668, places=4)
        self.assertAlmostEqual(results[1]['PSNR_score'], 60.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
