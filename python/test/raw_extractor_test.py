import unittest
import os

import numpy as np

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.raw_extractor import AssetExtractor, DisYUVRawVideoExtractor

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class RawExtractorTest(unittest.TestCase):

    def test_run_asset_extractor(self):

        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'quality_width': 160, 'quality_height': 90})

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324,
                                  'quality_width': 160, 'quality_height': 90})

        self.fextractor = AssetExtractor(
            [asset, asset_original], None, fifo_mode=True)

        self.fextractor.run()

        results = self.fextractor.results

        self.assertEqual(str(results[0]['asset']), 'test_0_1_src01_hrc00_576x324_576x324_vs_src01_hrc01_576x324_576x324_q_160x90')
        self.assertEqual(str(results[1]['asset']), 'test_0_2_src01_hrc00_576x324_576x324_vs_src01_hrc00_576x324_576x324_q_160x90')

        self.fextractor.run(parallelize=True)

        results = self.fextractor.results

        self.assertEqual(str(results[0]['asset']), 'test_0_1_src01_hrc00_576x324_576x324_vs_src01_hrc01_576x324_576x324_q_160x90')
        self.assertEqual(str(results[1]['asset']), 'test_0_2_src01_hrc00_576x324_576x324_vs_src01_hrc00_576x324_576x324_q_160x90')


class DisYUVRawVideoExtractorTest(unittest.TestCase):

    def setUp(self):
        self.h5py_filepath = VmafConfig.workdir_path('test.hdf5')

    def tearDown(self):
        if os.path.exists(self.h5py_filepath):
            os.remove(self.h5py_filepath)

    def test_run_dis_yuv_raw_video_extractor(self):
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324})

        h5py_file = DisYUVRawVideoExtractor.open_h5py_file(self.h5py_filepath)

        self.fextractor = DisYUVRawVideoExtractor(
            [asset, asset_original], None, fifo_mode=False,
            optional_dict={'channels': 'yu'},
            optional_dict2={'h5py_file': h5py_file}
        )

        self.fextractor.run()

        results = self.fextractor.results

        self.assertAlmostEqual(np.mean(results[0]['dis_y']), 61.332006579182384, places=4)
        self.assertAlmostEqual(np.mean(results[1]['dis_y']), 59.788567297525148, places=4)
        self.assertAlmostEqual(np.mean(results[0]['dis_u']), 115.23227407335962, places=4)
        self.assertAlmostEqual(np.mean(results[1]['dis_u']), 114.49701717535437, places=4)

        with self.assertRaises(KeyError):
            np.mean(results[0]['dis_v'])

        DisYUVRawVideoExtractor.close_h5py_file(h5py_file)

    def test_run_dis_yuv_raw_video_extractor_parallel(self):
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324})

        h5py_file = DisYUVRawVideoExtractor.open_h5py_file(self.h5py_filepath)

        self.fextractor = DisYUVRawVideoExtractor(
            [asset, asset_original], None, fifo_mode=False,
            optional_dict={'channels': 'yu'},
            optional_dict2={'h5py_file': h5py_file}
        )

        with self.assertRaises(AssertionError):
            self.fextractor.run(parallelize=True)

        DisYUVRawVideoExtractor.close_h5py_file(h5py_file)


class ParallelDisYRawVideoExtractorTest(unittest.TestCase):

    def setUp(self):
        self.h5py_filepath = VmafConfig.workdir_path('test.hdf5')

    def tearDown(self):
        if os.path.exists(self.h5py_filepath):
            os.remove(self.h5py_filepath)

    def test_run_parallel_dis_y_fextractor(self):
        ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width': 576, 'height': 324})

        asset_original = Asset(dataset="test", content_id=0, asset_id=2,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_path,
                      dis_path=ref_path,
                      asset_dict={'width': 576, 'height': 324})

        h5py_file = DisYUVRawVideoExtractor.open_h5py_file(self.h5py_filepath)
        optional_dict2 = {'h5py_file': h5py_file}

        fextractor = DisYUVRawVideoExtractor(
            [asset, asset_original],
            None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'channels': 'yu'},
            optional_dict2=optional_dict2
        )
        self.fextractors = [fextractor]
        fextractor.run(parallelize=False)  # Can't run parallel: can't pickle FileID objects
        results = fextractor.results

        self.assertAlmostEqual(np.mean(results[0]['dis_y']), 61.332006579182384, places=4)
        self.assertAlmostEqual(np.mean(results[1]['dis_y']), 59.788567297525148, places=4)
        self.assertAlmostEqual(np.mean(results[0]['dis_u']), 115.23227407335962, places=4)
        self.assertAlmostEqual(np.mean(results[1]['dis_u']), 114.49701717535437, places=4)

        with self.assertRaises(KeyError):
            np.mean(results[0]['dis_v'])

        DisYUVRawVideoExtractor.close_h5py_file(h5py_file)


if __name__ == '__main__':
    unittest.main(verbosity=2)
