import unittest

import numpy as np

from vmaf.config import VmafConfig
from vmaf.core.adm_dwt2_cy import adm_dwt2_cy
from vmaf.core.adm_dwt2_py import adm_dwt2_py
from vmaf.tools.reader import YuvReader


class AdmDwt2CyTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_adm_dwt2_cy(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[324, 576]).astype(np.float64)
        a, v, h, d = adm_dwt2_cy(x)
        self.assertEqual(a.shape, (162, 288))
        self.assertEqual(v.shape, (162, 288))
        self.assertEqual(h.shape, (162, 288))
        self.assertEqual(d.shape, (162, 288))
        self.assertAlmostEqual(np.float(np.std(a)), 73.94278958581368, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 73.61917114257812, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 73.30349892103438, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 73.19024658203125, places=5)

    def test_adm_dwt2_py(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[324, 576]).astype(np.float64)
        a, ds = adm_dwt2_py(x)
        h, v, d = ds
        self.assertEqual(a.shape, (162, 288))
        self.assertEqual(v.shape, (162, 288))
        self.assertEqual(h.shape, (162, 288))
        self.assertEqual(d.shape, (162, 288))
        self.assertAlmostEqual(np.float(np.std(a)), 73.8959273922819, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 73.69196319580078, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 73.53559112548828, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 73.52173614501953, places=5)

    def test_adm_dwt2_cy_small(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[36, 44]).astype(np.float64)
        a, v, h, d = adm_dwt2_cy(x)
        self.assertEqual(a.shape, (18, 22))
        self.assertEqual(v.shape, (18, 22))
        self.assertEqual(h.shape, (18, 22))
        self.assertEqual(d.shape, (18, 22))
        self.assertAlmostEqual(np.float(np.std(a)), 71.41875192644292, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 72.01703463816919, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 71.96445272697756, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 67.12256460863848, places=5)

    def test_adm_dwt2_cy_xsmall(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[18, 22]).astype(np.float64)
        a, v, h, d = adm_dwt2_cy(x)
        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))
        self.assertAlmostEqual(np.float(np.std(a)), 68.16857895299466, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 64.45100717633085, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 74.53569012139673, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 61.761051507492866, places=5)

    def test_adm_dwt2_cy_xsmallP_dc(self):
        x = (55 * np.ones([18, 22])).astype(np.float64)
        a, v, h, d = adm_dwt2_cy(x)

        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))

        self.assertAlmostEqual(np.float(np.max(a)), 109.99999999999997, places=16)
        self.assertAlmostEqual(np.float(np.max(v)), 8.526512829121202e-14, places=16)
        self.assertAlmostEqual(np.float(np.max(h)), 8.038873388460928e-14, places=16)
        self.assertAlmostEqual(np.float(np.max(d)), 0.0, places=16)

        self.assertAlmostEqual(np.float(np.min(a)), 0.0, places=16)
        self.assertAlmostEqual(np.float(np.min(v)), 0.0, places=16)
        self.assertAlmostEqual(np.float(np.min(h)), 0.0, places=16)
        self.assertAlmostEqual(np.float(np.min(d)), 0.0, places=16)

    def test_adm_dwt2_cy_dc(self):
        x = (55 * np.ones([324, 576])).astype(np.float64)
        a, v, h, d = adm_dwt2_cy(x)

        self.assertEqual(a.shape, (162, 288))
        self.assertEqual(v.shape, (162, 288))
        self.assertEqual(h.shape, (162, 288))
        self.assertEqual(d.shape, (162, 288))

        self.assertAlmostEqual(np.float(np.max(a)), 109.99999999999999, places=8)
        self.assertAlmostEqual(np.float(np.max(v)), 8.526512829121202e-14, places=16)
        self.assertAlmostEqual(np.float(np.max(h)), 8.038873388460928e-14, places=16)
        self.assertAlmostEqual(np.float(np.max(d)), 0.0, places=16)

        self.assertAlmostEqual(np.float(np.min(a)), 109.99999999999997, places=16)
        self.assertAlmostEqual(np.float(np.min(v)), 8.526512829121202e-14, places=16)
        self.assertAlmostEqual(np.float(np.min(h)), 8.038873388460928e-14, places=16)
        self.assertAlmostEqual(np.float(np.min(d)), 0.0, places=16)


class AdmDwt2CyTestOnAkiyo(unittest.TestCase):

    def setUp(self) -> None:
        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288"),
                width=352, height=288, yuv_type='yuv420p'
        ) as yuv_reader_ref:
            self.y_ref = yuv_reader_ref.next()[0].astype(np.float64)
        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288"),
                width=352, height=288, yuv_type='yuv420p'
        ) as yuv_reader_ref:
            self.y_dis = yuv_reader_ref.next()[0].astype(np.float64)

    def test_adm_dwt2_cy_on_akiyo_single_scale(self):

        a, v, h, d = adm_dwt2_cy(self.y_ref - 128.0)
        self.assertEqual(a.shape, (144, 176))
        self.assertEqual(v.shape, (144, 176))
        self.assertEqual(h.shape, (144, 176))
        self.assertEqual(d.shape, (144, 176))
        self.assertAlmostEqual(np.float(np.std(a)), 94.61946432328955, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 8.238639778464304, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 4.652637049444403, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 2.1727321628143614, places=5)
        self.assertAlmostEqual(np.float(np.mean(a)), -68.27941073300276, places=5)
        self.assertAlmostEqual(np.float(np.mean(v)), -0.05449070177725589, places=5)
        self.assertAlmostEqual(np.float(np.mean(h)), 0.004257626855952768, places=5)
        self.assertAlmostEqual(np.float(np.mean(d)), -0.002311283312114316, places=5)

        a, v, h, d = adm_dwt2_cy(self.y_dis - 128.0)
        self.assertEqual(a.shape, (144, 176))
        self.assertEqual(v.shape, (144, 176))
        self.assertEqual(h.shape, (144, 176))
        self.assertEqual(d.shape, (144, 176))
        self.assertAlmostEqual(np.float(np.std(a)), 108.79343831916911, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 9.51322612796857, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 5.4237937163381, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 2.6714361298075886, places=5)
        self.assertAlmostEqual(np.float(np.mean(a)), -41.14129780482415, places=4)
        self.assertAlmostEqual(np.float(np.mean(v)), -0.05419786586456123, places=5)
        self.assertAlmostEqual(np.float(np.mean(h)), 0.002907897856746483, places=5)
        self.assertAlmostEqual(np.float(np.mean(d)), -0.000920162342243558, places=5)
        self.assertAlmostEqual(np.float(np.max(a)), 246.64342088710973, places=5)
        self.assertAlmostEqual(np.float(np.max(v)), 153.63339344423332, places=5)
        self.assertAlmostEqual(np.float(np.max(h)), 120.05215038354704, places=5)
        self.assertAlmostEqual(np.float(np.max(d)), 113.9025450046058, places=5)
        self.assertAlmostEqual(np.float(np.min(a)), -269.44402359826955, places=4)
        self.assertAlmostEqual(np.float(np.min(v)), -128.21006288062947, places=5)
        self.assertAlmostEqual(np.float(np.min(h)), -101.95207793605813, places=5)
        self.assertAlmostEqual(np.float(np.min(d)), -51.79502250236081, places=5)

    def test_adm_dwt2_cy_on_akiyo_n_scales(self):

        a = self.y_dis - 128.0

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (144, 176))
        self.assertEqual(v.shape, (144, 176))
        self.assertEqual(h.shape, (144, 176))
        self.assertEqual(d.shape, (144, 176))
        self.assertAlmostEqual(np.float(np.std(a)), 108.79343831916911, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 9.51322612796857, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 5.4237937163381, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 2.6714361298075886, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (72, 88))
        self.assertEqual(v.shape, (72, 88))
        self.assertEqual(h.shape, (72, 88))
        self.assertEqual(d.shape, (72, 88))
        self.assertAlmostEqual(np.float(np.std(a)), 214.13138192032943, places=4)
        self.assertAlmostEqual(np.float(np.std(v)), 25.622761528555454, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 17.70038682132729, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 11.180386017176657, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (36, 44))
        self.assertEqual(v.shape, (36, 44))
        self.assertEqual(h.shape, (36, 44))
        self.assertEqual(d.shape, (36, 44))
        self.assertAlmostEqual(np.float(np.std(a)), 417.73602985019386, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 56.12149501761274, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 45.419327754665, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 23.307968058054456, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (18, 22))
        self.assertEqual(v.shape, (18, 22))
        self.assertEqual(h.shape, (18, 22))
        self.assertEqual(d.shape, (18, 22))
        self.assertAlmostEqual(np.float(np.std(a)), 743.8105722045989, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 116.11048356419965, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 112.95438137920729, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 45.10849973956695, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))
        self.assertAlmostEqual(np.float(np.std(a)), 1086.6252584211409, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 231.09377461005127, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 562.4967265458288, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 324.7682467339999, places=5)


class AdmDwt2CyTestOnAkiyoXsmall(unittest.TestCase):

    def setUp(self) -> None:
        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_18x22"),
                width=22, height=18, yuv_type='yuv420p'
        ) as yuv_reader_ref:
            self.y_ref = yuv_reader_ref.next()[0].astype(np.float64)
        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_18x22"),
                width=22, height=18, yuv_type='yuv420p'
        ) as yuv_reader_ref:
            self.y_dis = yuv_reader_ref.next()[0].astype(np.float64)

    def test_adm_dwt2_cy_on_akiyo_single_scale(self):

        a, v, h, d = adm_dwt2_cy((self.y_ref - 128.0).astype(np.float64))
        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))
        self.assertAlmostEqual(np.float(np.std(a)), 68.79884774616018, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 13.512513212683404, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 44.15490291235366, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 17.07667660487361, places=5)
        self.assertAlmostEqual(np.float(np.mean(a)), -75.92569837539651, places=5)
        self.assertAlmostEqual(np.float(np.mean(v)), -3.5294110328601627, places=5)
        self.assertAlmostEqual(np.float(np.mean(h)), 1.1199154418499353, places=5)
        self.assertAlmostEqual(np.float(np.mean(d)), -0.48905969923597864, places=5)

        self.assertAlmostEqual(a[0][0], -145.9108983848698, places=6)
        self.assertAlmostEqual(a[0][-1], -140.54543100196207, places=6)
        self.assertAlmostEqual(a[1][0], 0.0, places=6)
        self.assertAlmostEqual(a[1][-1], -103.95207793609761, places=6)
        self.assertAlmostEqual(a[-1][0], 106.86207299548309, places=5)
        self.assertAlmostEqual(a[-1][-1], 50.15606578219574, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
