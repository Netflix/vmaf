__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.vqm_quality_runner import VqmGeneralQualityRunner, VqmVfdQualityRunner

class VqmQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()

    def test_run_vqm_general_runner(self):
        print 'test on running VQM general runner...'
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324_x2.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324_x2.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':288, 'quality_height':162,
                                  'start_frame':0, 'end_frame':95, 'fps':20, })

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':288, 'quality_height':162,
                                  'start_frame':0, 'end_frame':95, 'fps':20, })

        self.runner = VqmGeneralQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VQM_General_score'], 0.786717, places=4)
        self.assertAlmostEqual(results[1]['VQM_General_score'], 1.0, places=4)

    def test_run_vqm_vfd_runner(self):
        print 'test on running VQM VFD runner...'
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324_x2.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324_x2.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':288, 'quality_height':162,
                                  'start_frame':0, 'end_frame':95, 'fps':20, })

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':288, 'quality_height':162,
                                  'start_frame':0, 'end_frame':95, 'fps':20, })

        self.runner = VqmVfdQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run()

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VQM_VFD_score'], 0.854375, places=4)
        self.assertAlmostEqual(results[1]['VQM_VFD_score'], 0.935739, places=4)


class ParallelQualityRunnerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'runners'):
            for runner in self.runners:
                runner.remove_results()

    def test_run_parallel_vqm_general_runner(self):
        print 'test on running VQM general quality runner in parallel...'
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324_x2.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324_x2.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':288, 'quality_height':162,
                                  'start_frame':0, 'end_frame':95, 'fps':20, })

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':288, 'quality_height':162,
                                  'start_frame':0, 'end_frame':95, 'fps':20, })

        self.runner = VqmGeneralQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VQM_General_score'], 0.786717, places=4)
        self.assertAlmostEqual(results[1]['VQM_General_score'], 1.0, places=4)

    def test_run_vqm_vfd_runner(self):
        print 'test on running VQM VFD runner in parallel...'
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324_x2.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324_x2.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':288, 'quality_height':162,
                                  'start_frame':0, 'end_frame':95, 'fps':20, })

        asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width':576, 'height':324,
                                  'quality_width':288, 'quality_height':162,
                                  'start_frame':0, 'end_frame':95, 'fps':20, })

        self.runner = VqmVfdQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            delete_workdir=True,
            result_store=None
        )
        self.runner.run(parallelize=True)

        results = self.runner.results
        self.assertAlmostEqual(results[0]['VQM_VFD_score'], 0.854375, places=4)
        self.assertAlmostEqual(results[1]['VQM_VFD_score'], 0.935739, places=4)

if __name__ == '__main__':
    unittest.main()
