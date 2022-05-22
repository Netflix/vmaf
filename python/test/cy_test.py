import unittest

import numpy as np

from vmaf.config import VmafConfig
from vmaf.core.adm_dwt2_cy import adm_dwt2_cy
from vmaf.core.adm_dwt2_py import adm_dwt2_py, adm_idwt2_py
from vmaf.tools.reader import YuvReader


class AdmDwt2PyTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_adm_dwt2_py(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[324, 576]).astype(np.float64)
        a, v, h, d = adm_dwt2_py(x)
        self.assertEqual(a.shape, (162, 288))
        self.assertEqual(v.shape, (162, 288))
        self.assertEqual(h.shape, (162, 288))
        self.assertEqual(d.shape, (162, 288))
        self.assertAlmostEqual(float(np.std(a)), 73.8959273922819, places=5)
        self.assertAlmostEqual(float(np.std(v)), 73.69196319580078, places=5)
        self.assertAlmostEqual(float(np.std(h)), 73.53559112548828, places=5)
        self.assertAlmostEqual(float(np.std(d)), 73.52173614501953, places=5)

    def test_adm_dwt2_idwt2_py(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[324, 576]).astype(np.float64)
        a, v, h, d = adm_dwt2_py(x)
        x2 = adm_idwt2_py((a, v, h, d))
        self.assertAlmostEqual(np.abs(np.max(x - x2)), 0.0, places=10)


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
        self.assertAlmostEqual(float(np.std(a)), 73.94278958581368, places=5)
        self.assertAlmostEqual(float(np.std(v)), 73.61917114257812, places=5)
        self.assertAlmostEqual(float(np.std(h)), 73.30349892103438, places=5)
        self.assertAlmostEqual(float(np.std(d)), 73.19024658203125, places=5)

    def test_adm_dwt2_cy_small(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[36, 44]).astype(np.float64)
        a, v, h, d = adm_dwt2_cy(x)
        self.assertEqual(a.shape, (18, 22))
        self.assertEqual(v.shape, (18, 22))
        self.assertEqual(h.shape, (18, 22))
        self.assertEqual(d.shape, (18, 22))
        self.assertAlmostEqual(float(np.std(a)), 75.67939583965345, places=5)
        self.assertAlmostEqual(float(np.std(v)), 74.84364345847594, places=5)
        self.assertAlmostEqual(float(np.std(h)), 73.1595099161633, places=5)
        self.assertAlmostEqual(float(np.std(d)), 68.17639102914511, places=5)

    def test_adm_dwt2_cy_xsmall(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[18, 22]).astype(np.float64)
        a, v, h, d = adm_dwt2_cy(x)
        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))
        self.assertAlmostEqual(float(np.std(a)), 68.79495062113115, places=5)
        self.assertAlmostEqual(float(np.std(v)), 67.96417675879844, places=5)
        self.assertAlmostEqual(float(np.std(h)), 75.05987575325626, places=5)
        self.assertAlmostEqual(float(np.std(d)), 62.93524366183755, places=5)

    def test_adm_dwt2_cy_xsmallP_dc(self):
        x = (55 * np.ones([18, 22])).astype(np.float64)
        a, v, h, d = adm_dwt2_cy(x)

        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))

        self.assertAlmostEqual(float(np.max(a)), 109.99999999999997, places=16)
        self.assertAlmostEqual(float(np.max(v)), 8.526512829121202e-14, places=16)
        self.assertAlmostEqual(float(np.max(h)), 8.038873388460928e-14, places=16)
        self.assertAlmostEqual(float(np.max(d)), 0.0, places=16)

        self.assertAlmostEqual(float(np.min(a)), 109.99999999999997, places=16)
        self.assertAlmostEqual(float(np.min(v)), 8.526512829121202e-14, places=16)
        self.assertAlmostEqual(float(np.min(h)), 8.038873388460928e-14, places=16)
        self.assertAlmostEqual(float(np.min(d)), 0.0, places=16)

    def test_adm_dwt2_cy_dc(self):
        x = (55 * np.ones([324, 576])).astype(np.float64)
        a, v, h, d = adm_dwt2_cy(x)

        self.assertEqual(a.shape, (162, 288))
        self.assertEqual(v.shape, (162, 288))
        self.assertEqual(h.shape, (162, 288))
        self.assertEqual(d.shape, (162, 288))

        self.assertAlmostEqual(float(np.max(a)), 109.99999999999999, places=8)
        self.assertAlmostEqual(float(np.max(v)), 8.526512829121202e-14, places=16)
        self.assertAlmostEqual(float(np.max(h)), 8.038873388460928e-14, places=16)
        self.assertAlmostEqual(float(np.max(d)), 0.0, places=16)

        self.assertAlmostEqual(float(np.min(a)), 109.99999999999997, places=16)
        self.assertAlmostEqual(float(np.min(v)), 8.526512829121202e-14, places=16)
        self.assertAlmostEqual(float(np.min(h)), 8.038873388460928e-14, places=16)
        self.assertAlmostEqual(float(np.min(d)), 0.0, places=16)


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
        self.assertAlmostEqual(float(np.std(a)), 94.61946432328955, places=5)
        self.assertAlmostEqual(float(np.std(v)), 8.238639778464304, places=5)
        self.assertAlmostEqual(float(np.std(h)), 4.652637049444403, places=5)
        self.assertAlmostEqual(float(np.std(d)), 2.1727321628143614, places=5)
        self.assertAlmostEqual(float(np.mean(a)), -68.27941073300276, places=5)
        self.assertAlmostEqual(float(np.mean(v)), -0.05449070177725589, places=5)
        self.assertAlmostEqual(float(np.mean(h)), 0.004257626855952768, places=5)
        self.assertAlmostEqual(float(np.mean(d)), -0.002311283312114316, places=5)

        a, v, h, d = adm_dwt2_cy(self.y_dis - 128.0)
        self.assertEqual(a.shape, (144, 176))
        self.assertEqual(v.shape, (144, 176))
        self.assertEqual(h.shape, (144, 176))
        self.assertEqual(d.shape, (144, 176))
        self.assertAlmostEqual(float(np.std(a)), 108.79343831916911, places=5)
        self.assertAlmostEqual(float(np.std(v)), 9.51322612796857, places=5)
        self.assertAlmostEqual(float(np.std(h)), 5.4237937163381, places=5)
        self.assertAlmostEqual(float(np.std(d)), 2.6714361298075886, places=5)
        self.assertAlmostEqual(float(np.mean(a)), -41.14129780482415, places=4)
        self.assertAlmostEqual(float(np.mean(v)), -0.05419786586456123, places=5)
        self.assertAlmostEqual(float(np.mean(h)), 0.002907897856746483, places=5)
        self.assertAlmostEqual(float(np.mean(d)), -0.000920162342243558, places=5)
        self.assertAlmostEqual(float(np.max(a)), 246.64342088710973, places=5)
        self.assertAlmostEqual(float(np.max(v)), 153.63339344423332, places=5)
        self.assertAlmostEqual(float(np.max(h)), 120.05215038354704, places=5)
        self.assertAlmostEqual(float(np.max(d)), 113.9025450046058, places=5)
        self.assertAlmostEqual(float(np.min(a)), -269.44402359826955, places=4)
        self.assertAlmostEqual(float(np.min(v)), -128.21006288062947, places=5)
        self.assertAlmostEqual(float(np.min(h)), -101.95207793605813, places=5)
        self.assertAlmostEqual(float(np.min(d)), -51.79502250236081, places=5)

    def test_adm_dwt2_cy_on_akiyo_n_scales(self):

        a = self.y_dis - 128.0

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (144, 176))
        self.assertEqual(v.shape, (144, 176))
        self.assertEqual(h.shape, (144, 176))
        self.assertEqual(d.shape, (144, 176))
        self.assertAlmostEqual(float(np.std(a)), 108.79343831916911, places=5)
        self.assertAlmostEqual(float(np.std(v)), 9.51322612796857, places=5)
        self.assertAlmostEqual(float(np.std(h)), 5.4237937163381, places=5)
        self.assertAlmostEqual(float(np.std(d)), 2.6714361298075886, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (72, 88))
        self.assertEqual(v.shape, (72, 88))
        self.assertEqual(h.shape, (72, 88))
        self.assertEqual(d.shape, (72, 88))
        self.assertAlmostEqual(float(np.std(a)), 214.13138192032943, places=4)
        self.assertAlmostEqual(float(np.std(v)), 25.622761528555454, places=5)
        self.assertAlmostEqual(float(np.std(h)), 17.70038682132729, places=5)
        self.assertAlmostEqual(float(np.std(d)), 11.180386017176657, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (36, 44))
        self.assertEqual(v.shape, (36, 44))
        self.assertEqual(h.shape, (36, 44))
        self.assertEqual(d.shape, (36, 44))
        self.assertAlmostEqual(float(np.std(a)), 417.73602985019386, places=5)
        self.assertAlmostEqual(float(np.std(v)), 56.12149501761274, places=5)
        self.assertAlmostEqual(float(np.std(h)), 45.419327754665, places=5)
        self.assertAlmostEqual(float(np.std(d)), 23.307968058054456, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (18, 22))
        self.assertEqual(v.shape, (18, 22))
        self.assertEqual(h.shape, (18, 22))
        self.assertEqual(d.shape, (18, 22))
        self.assertAlmostEqual(float(np.std(a)), 800.3512673377832, places=5)
        self.assertAlmostEqual(float(np.std(v)), 133.36092909420165, places=5)
        self.assertAlmostEqual(float(np.std(h)), 113.7904328069727, places=5)
        self.assertAlmostEqual(float(np.std(d)), 47.651801810101496, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))
        self.assertAlmostEqual(float(np.std(a)), 1477.3950051168213, places=5)
        self.assertAlmostEqual(float(np.std(v)), 284.3777998557365, places=5)
        self.assertAlmostEqual(float(np.std(h)), 324.15282224075446, places=5)
        self.assertAlmostEqual(float(np.std(d)), 116.24440187217311, places=5)


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
        self.assertAlmostEqual(float(np.std(a)), 70.96065160084203, places=5)
        self.assertAlmostEqual(float(np.std(v)), 14.048580053657007, places=5)
        self.assertAlmostEqual(float(np.std(h)), 44.79487471014486, places=5)
        self.assertAlmostEqual(float(np.std(d)), 17.35515294883506, places=5)
        self.assertAlmostEqual(float(np.mean(a)), -75.84326395274935, places=5)
        self.assertAlmostEqual(float(np.mean(v)), -3.2531880028079962, places=5)
        self.assertAlmostEqual(float(np.mean(h)), 1.818506248770275, places=5)
        self.assertAlmostEqual(float(np.mean(d)), -0.3250551358064137, places=5)

        self.assertAlmostEqual(a[0][0], -145.9108983848698, places=6)
        self.assertAlmostEqual(a[0][-1], -140.54543100196207, places=6)
        self.assertAlmostEqual(a[1][0], -136.4577676021147, places=6)
        self.assertAlmostEqual(a[1][-1], -120.36561690415635, places=6)
        self.assertAlmostEqual(a[-1][0], 79.6265877365092, places=5)
        self.assertAlmostEqual(a[-1][-1], 44.29716825973104, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
