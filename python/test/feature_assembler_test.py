__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

from vmaf.core.feature_assembler import FeatureAssembler
from vmaf.core.feature_extractor import VmafFeatureExtractor, FeatureExtractor, \
    MomentFeatureExtractor
from testutil import set_default_576_324_videos_for_testing


class FeatureAssemblerTest(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'fassembler'):
            self.fassembler.remove_results()
        pass

    def test_get_fextractor_subclasses(self):
        fextractor_subclasses = FeatureExtractor.get_subclasses_recursively()
        self.assertTrue(VmafFeatureExtractor in fextractor_subclasses)
        self.assertTrue(MomentFeatureExtractor in fextractor_subclasses)

    def test_feature_assembler_whole_feature(self):
        print 'test on feature assembler with whole feature...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fassembler = FeatureAssembler(
            feature_dict={'VMAF_feature':'all'},
            feature_option_dict=None,
            assets=[asset, asset_original],
            logger=None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=None,
            parallelize=True,
        )

        self.fassembler.run()

        results = self.fassembler.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.93458780728708746, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833333, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

    def test_feature_assembler_selected_atom_feature(self):
        print 'test on feature assembler with selected atom features...'
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fassembler = FeatureAssembler(
            feature_dict={'VMAF_feature':['vif', 'motion']},
            feature_option_dict=None,
            assets=[asset, asset_original],
            logger=None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=None,
            parallelize=True,
        )

        self.fassembler.run()

        results = self.fassembler.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)

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
