from __future__ import absolute_import

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import unittest

from vmaf.core.feature_assembler import FeatureAssembler
from vmaf.core.feature_extractor import VmafFeatureExtractor, FeatureExtractor, \
    MomentFeatureExtractor

from test.testutil import set_default_576_324_videos_for_testing


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

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fassembler = FeatureAssembler(
            feature_dict={'VMAF_feature': 'all'},
            feature_option_dict=None,
            assets=[asset, asset_original],
            logger=None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=None,
            parallelize=True,
            processes=None,
        )

        self.fassembler.run()

        results = self.fassembler.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833333, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

    def test_feature_assembler_whole_feature_processes(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fassembler = FeatureAssembler(
            feature_dict={'VMAF_feature': 'all'},
            feature_option_dict=None,
            assets=[asset, asset_original],
            logger=None,
            fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict=None,
            optional_dict2=None,
            parallelize=True,
            processes=1,
        )

        self.fassembler.run()

        results = self.fassembler.results

        self.assertAlmostEqual(results[0]['VMAF_feature_vif_score'], 0.44609306249999997, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_adm2_score'], 0.9345149030293786, places=4)
        self.assertAlmostEqual(results[0]['VMAF_feature_ansnr_score'], 23.509571520833333, places=4)

        self.assertAlmostEqual(results[1]['VMAF_feature_vif_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_motion_score'], 4.0498253541666669, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_adm2_score'], 1.0, places=4)
        self.assertAlmostEqual(results[1]['VMAF_feature_ansnr_score'], 31.271439270833337, places=4)

    def test_feature_assembler_selected_atom_feature(self):

        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        self.fassembler = FeatureAssembler(
            feature_dict={'VMAF_feature': ['vif', 'motion']},
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
            _ = results[0]['VMAF_feature_ansnr_scores']
        with self.assertRaises(KeyError):
            _ = results[0]['VMAF_feature_ansnr_score']
        with self.assertRaises(KeyError):
            _ = results[0]['VMAF_feature_adm_scores']
        with self.assertRaises(KeyError):
            _ = results[0]['VMAF_feature_adm_score']


class FeatureAssemblerUnitTest(unittest.TestCase):

    def test_feature_assembler_get_fextractor_instance(self):
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()

        fassembler = FeatureAssembler(
            feature_dict={'VMAF_feature': ['vif_scale0', 'vif_scale1', 'vif_scale2', 'vif_scale3', 'adm2', 'motion2']},
            feature_option_dict={'VMAF_feature': {'adm_ref_display_height': 540}},
            assets=[asset],
            logger=None,
            fifo_mode=False,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': 'model/vmaf_float_v0.6.1_rdh540.json'},
            optional_dict2=None,
            parallelize=False,
            save_workfiles=False,
        )

        fex = fassembler._get_fextractor_instance('VMAF_feature')
        self.assertEqual(fex.optional_dict, {'adm_ref_display_height': 540})
        self.assertEqual(fex.optional_dict2, None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
