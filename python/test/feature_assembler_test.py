__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

from core.asset import Asset
import config
from core.feature_assembler import FeatureAssembler
from core.feature_extractor import VmafFeatureExtractor, FeatureExtractor, \
    MomentFeatureExtractor


class FeatureAssemblerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fassembler'):
            self.fassembler.remove_results()
        pass

    def test_get_fextractor_subclasses(self):
        fextractor_subclasses = FeatureExtractor.get_subclasses_recursively()
        self.assertEquals(len(fextractor_subclasses), 5)
        self.assertTrue(VmafFeatureExtractor in fextractor_subclasses)
        self.assertTrue(MomentFeatureExtractor in fextractor_subclasses)

    def test_feature_assembler_whole_feature(self):
        print 'test on feature assembler with whole feature...'
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

        self.fassembler = FeatureAssembler(
            feature_dict = {'VMAF_feature':'all'},
            feature_option_dict = None,
            assets = [asset, asset_original],
            logger=None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            parallelize=True,
        )

        self.fassembler.run()

        results = self.fassembler.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44455808333333313, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9254334398006141, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 22.533456770833329, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 30.030914145833322, places=4)

    def test_feature_assembler_selected_atom_feature(self):
        print 'test on feature assembler with selected atom features...'
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

        self.fassembler = FeatureAssembler(
            feature_dict = {'VMAF_feature':['vif', 'motion']},
            feature_option_dict = None,
            assets = [asset, asset_original],
            logger=None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            parallelize=True,
        )

        self.fassembler.run()

        results = self.fassembler.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44455808333333313, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 3.5916076041666667, places=4)

        with self.assertRaises(KeyError):
            results[0]['VMAF_feature_ansnr_scores']
        with self.assertRaises(KeyError):
            results[0]['VMAF_feature_ansnr_score']
        with self.assertRaises(KeyError):
            results[0]['VMAF_feature_adm_scores']
        with self.assertRaises(KeyError):
            results[0]['VMAF_feature_adm_score']



if __name__ == '__main__':
    unittest.main()
