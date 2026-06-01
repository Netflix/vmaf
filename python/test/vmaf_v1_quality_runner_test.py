from __future__ import absolute_import

import unittest

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafexecQualityRunner
from vmaf.tools.misc import MyTestCase

from test.testutil import set_default_576_324_videos_for_testing

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class VmafV1QualityRunnerTest(MyTestCase):

    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.remove_results()
        super().tearDown()

    def test_v1016_integer_1080_3d0H_exec(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v1.0.16', 'vmaf_v1.0.16_3d0h.json')}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0][
                                   'VMAFEXEC_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_score'],
                               0.9482851666666666, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion3_mmxv_18_score'], 3.9897647708333337, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_324_encw_576_score'],
                               0.2596781458333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5_score'],
                               6.141235312500001, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 82.81605935416667, places=4)

    def test_v1016_integer_hfr_1080_3d0H_exec(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v1.0.16_hfr', 'vmaf_v1.0.16_hfr_3d0h.json')}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0][
                                   'VMAFEXEC_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_score'],
                               0.9482851666666666, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion3_mffw_mmxv_18_mma_score'], 6.828103625, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_324_encw_576_score'],
                               0.2596781458333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5_score'],
                               6.141235312500001, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 84.802363, places=4)

    def test_v1016_integer_2160_3d0H_exec(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v1.0.16', 'vmaf_v1.0.16_3d0h_2160.json')}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160_score'], 0.9595838749999999, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion3_mmxv_18_score'], 3.9897647708333337, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_324_encw_576_score'], 0.2596781458333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5_score'], 0.0, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 92.26304791666666, places=4)

    def test_v1016_integer_hfr_2160_3d0H_exec(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v1.0.16_hfr', 'vmaf_v1.0.16_hfr_3d0h_2160.json')}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_rdh_2160_score'], 0.9595838749999999, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion3_mffw_mmxv_18_mma_score'], 6.828103625, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_324_encw_576_score'], 0.2596781458333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_speed_chroma_uv_mxv_45_nnf_0.1_ps_0.5_psm_bilinear_snn_0.19_wvm_5_score'], 0.0, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 94.45015866666665, places=4)

    def test_v1016_integer_2160_1d5H_exec(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v1.0.16', 'vmaf_v1.0.16_1d5h_2160.json')}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160_score'], 0.9482851666666666, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion3_mmxv_18_score'], 3.9897658541666665, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_324_encw_576_score'], 0.2596781458333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5_score'], 6.141236604166667, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 82.43278902083334, places=4)

    def test_v1016_integer_1080_5d0H_exec(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v1.0.16', 'vmaf_v1.0.16_5d0h.json')}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5_score'], 0.9564286041666666, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion3_mmxv_18_score'], 3.9897658541666665, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_324_encw_576_score'], 0.2596781458333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5_score'], 0.0, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 85.9211639375, places=4)

    def test_v1016_integer_hfr_2160_1d5H_exec(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v1.0.16_hfr', 'vmaf_v1.0.16_hfr_1d5h_2160.json')}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_1.5_rdh_2160_score'], 0.9482851666666666, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion3_mffw_mmxv_18_mma_score'], 6.828103333333334, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_324_encw_576_score'], 0.2596781458333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_speed_chroma_uv_mxv_45_nnf_0.1_snn_0.19_wvm_5_score'], 6.141236604166667, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 84.64331647916667, places=4)

    def test_v1016_integer_hfr_1080_5d0H_exec(self):
        _, _, asset, asset_original = set_default_576_324_videos_for_testing()
        self.runner = VmafexecQualityRunner(
            [asset, asset_original],
            None, fifo_mode=True,
            delete_workdir=True,
            result_store=None,
            optional_dict={'model_filepath': VmafConfig.model_path('vmaf_v1.0.16_hfr', 'vmaf_v1.0.16_hfr_5d0h.json')}
        )
        self.runner.run()

        results = self.runner.results

        self.assertAlmostEqual(results[0]['VMAFEXEC_adm3_csf_2_dlmw_0.7_egl_1_min_0.5_nw_0.02_nvd_5_score'], 0.9564286041666666, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_motion3_mffw_mmxv_18_mma_score'], 6.828103333333334, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_324_encw_576_score'], 0.2596781458333333, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_speed_chroma_uv_mxv_45_nnf_0.1_ps_0.6_psm_bilinear_snn_0.19_wvm_5_score'], 0.0, places=5)
        self.assertAlmostEqual(results[0]['VMAFEXEC_score'], 87.88518320833333, places=4)

    def test_v1016_integer_1080_3d0H_cambi_enc_override_exec(self):
        # The model's CAMBI feature accepts enc_width/enc_height/enc_bitdepth
        # overrides (the encode resolution/bitdepth before any scaling was
        # applied), passed through the asset's dis_enc_width/dis_enc_height/
        # dis_enc_bitdepth. This confirms (a) the overrides flow into the model's
        # single CAMBI instance -- the feature key carries the merged values and
        # the default key is no longer present (no duplicate registration) -- and
        # (b) they genuinely change the CAMBI score used in prediction.
        ref = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
        dis = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
        model = VmafConfig.model_path('vmaf_v1.0.16', 'vmaf_v1.0.16_3d0h.json')

        def run(extra):
            dis_dict = {'width': 576, 'height': 324}
            dis_dict.update(extra)
            asset = Asset(dataset="test", content_id=0, asset_id=0,
                          workdir_root=VmafConfig.workdir_path(),
                          ref_path=ref, dis_path=dis, asset_dict=dis_dict)
            asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                                   workdir_root=VmafConfig.workdir_path(),
                                   ref_path=ref, dis_path=ref,
                                   asset_dict={'width': 576, 'height': 324})
            self.runner = VmafexecQualityRunner(
                [asset, asset_original], None, fifo_mode=True, delete_workdir=True,
                result_store=None, optional_dict={'model_filepath': model})
            self.runner.run()
            return self.runner.results[0]

        default_key = 'VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_324_encw_576_score'

        # default: enc_* falls back to the 576x324 8-bit input dimensions
        r = run({})
        keys = r.get_ordered_list_score_key()
        self.assertIn(default_key, keys)
        self.assertEqual(len([k for k in keys if 'cambi' in k]), 1)
        default_cambi = r[default_key]
        self.assertAlmostEqual(default_cambi, 0.2596781458333333, places=5)
        self.assertAlmostEqual(r['VMAFEXEC_score'], 82.81606012499999, places=4)

        # encode-resolution override (320x180): the CAMBI key reflects the
        # override, the default key is gone, and the score changes
        r = run({'dis_enc_width': 320, 'dis_enc_height': 180})
        keys = r.get_ordered_list_score_key()
        override_key = 'VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_8_ench_180_encw_320_score'
        self.assertIn(override_key, keys)
        self.assertNotIn(default_key, keys)
        self.assertEqual(len([k for k in keys if 'cambi' in k]), 1)
        self.assertAlmostEqual(r[override_key], 0.0642141875, places=5)
        self.assertNotAlmostEqual(r[override_key], default_cambi, places=3)
        self.assertAlmostEqual(r['VMAFEXEC_score'], 82.9629780625, places=4)

        # encode-bitdepth override (10): same behaviour via enc_bitdepth
        r = run({'dis_enc_bitdepth': 10})
        keys = r.get_ordered_list_score_key()
        override_key = 'VMAFEXEC_cambi_hrs_1080_cmxv_17_vlt_0.06_encbd_10_ench_324_encw_576_score'
        self.assertIn(override_key, keys)
        self.assertNotIn(default_key, keys)
        self.assertEqual(len([k for k in keys if 'cambi' in k]), 1)
        self.assertAlmostEqual(r[override_key], 1.0104293958333332, places=5)
        self.assertNotAlmostEqual(r[override_key], default_cambi, places=3)
        self.assertAlmostEqual(r['VMAFEXEC_score'], 82.25588416666666, places=4)


if __name__ == '__main__':
    unittest.main()
