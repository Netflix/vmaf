import unittest
from vmaf.config import VmafConfig
from vmaf.routine import read_dataset
from vmaf.tools.misc import import_python_file

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class TestReadDataset(unittest.TestCase):

    def test_read_dataset_crop_and_pad(self):
        train_dataset_path = VmafConfig.test_resource_path('example_dataset_crop_pad.py')
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        train_assets[0].asset_dict['ref_crop_cmd'] = '288:162:144:81'
        train_assets[1].asset_dict['ref_pad_cmd'] = 'iw+100:ih+100:50:50'
        train_assets[2].asset_dict['ref_crop_cmd'] = '288:162:144:81'
        train_assets[2].asset_dict['ref_pad_cmd'] = 'iw+288:ih+162:144:81'

        self.assertEqual(len(train_assets), 3)
        self.assertEqual(str(train_assets[0]), "example_0_1_src01_hrc00_576x324_576x324_crop288:162:144:81_vs_src01_hrc01_576x324_576x324_crop288:162:144:81_q_576x324")
        self.assertEqual(str(train_assets[1]), "example_0_2_src01_hrc00_576x324_576x324_padiw+100:ih+100:50:50_vs_src01_hrc01_576x324_576x324_padiw+100:ih+100:50:50_q_576x324")
        self.assertEqual(str(train_assets[2]), "example_0_3_src01_hrc00_576x324_576x324_crop288:162:144:81_padiw+288:ih+162:144:81_vs_src01_hrc01_576x324_576x324_crop288:162:144:81_padiw+288:ih+162:144:81_q_576x324")

    def test_read_dataset_duration_sec(self):
        train_dataset_path = VmafConfig.test_resource_path('example_dataset_crop_pad_duration_sec.py')
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertEqual(len(train_assets), 3)
        self.assertEqual(train_assets[0].asset_dict['duration_sec'], 5.0)
        self.assertEqual(train_assets[1].asset_dict['duration_sec'], 5.0)
        self.assertEqual(train_assets[2].asset_dict['duration_sec'], 5.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
