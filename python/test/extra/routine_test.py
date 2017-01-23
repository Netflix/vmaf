import unittest
import config
from routine import read_dataset
from tools.misc import import_python_file

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class TestReadDataset(unittest.TestCase):

    def test_read_dataset_crop_and_pad(self):
        train_dataset_path = config.ROOT + '/python/test/resource/example_dataset_crop_pad.py'
        train_dataset = import_python_file(train_dataset_path)
        train_assets = read_dataset(train_dataset)

        self.assertEquals(len(train_assets), 3)
        self.assertEquals(str(train_assets[0]), "example_0_1_src01_hrc00_576x324_576x324_vs_src01_hrc01_576x324_576x324_q_576x324_crop288:162:144:81")
        self.assertEquals(str(train_assets[1]), "example_0_2_src01_hrc00_576x324_576x324_vs_src01_hrc01_576x324_576x324_q_576x324_padiw+100:ih+100:50:50")
        self.assertEquals(str(train_assets[2]), "example_0_3_src01_hrc00_576x324_576x324_vs_src01_hrc01_576x324_576x324_q_576x324_crop288:162:144:81_padiw+288:ih+162:144:81")

