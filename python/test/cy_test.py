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
        x = np.random.uniform(low=-128, high=127, size=[324, 576]).astype(np.float32)
        a, v, h, d = adm_dwt2_cy(x)
        self.assertEqual(a.shape, (162, 288))
        self.assertEqual(v.shape, (162, 288))
        self.assertEqual(h.shape, (162, 288))
        self.assertEqual(d.shape, (162, 288))
        self.assertAlmostEqual(np.float(np.std(a)), 73.94279479980469, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 73.61917114257812, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 73.30350494384766, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 73.19024658203125, places=5)

    def test_adm_dwt2_py(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[324, 576]).astype(np.float32)
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

    @unittest.skip
    def test_adm_dwt2_cy_small(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[36, 44]).astype(np.float32)
        a, v, h, d = adm_dwt2_cy(x)
        self.assertEqual(a.shape, (18, 22))
        self.assertEqual(v.shape, (18, 22))
        self.assertEqual(h.shape, (18, 22))
        self.assertEqual(d.shape, (18, 22))
        self.assertAlmostEqual(np.float(np.std(a)), 71.41875192644292, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 72.01703463816919, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 71.96445272697756, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 67.12256460863848, places=5)

    @unittest.skip
    def test_adm_dwt2_cy_xsmall(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[18, 22]).astype(np.float32)
        a, v, h, d = adm_dwt2_cy(x)
        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))
        self.assertAlmostEqual(np.float(np.std(a)), 68.16857895299466, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 64.45100717633085, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 74.53569012139673, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 61.761051507492866, places=5)


class AdmDwt2CyTestOnAkiyo(unittest.TestCase):

    def setUp(self) -> None:
        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288"),
                width=352, height=288, yuv_type='yuv420p'
        ) as yuv_reader_ref:
            self.y_ref = yuv_reader_ref.next()[0].astype(np.float32)
        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_352x288"),
                width=352, height=288, yuv_type='yuv420p'
        ) as yuv_reader_ref:
            self.y_dis = yuv_reader_ref.next()[0].astype(np.float32)

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
        self.assertAlmostEqual(np.float(np.std(a)), 388.2596130371094, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 49.79795455932617, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 44.297210693359375, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 21.884857177734375, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (18, 22))
        self.assertEqual(v.shape, (18, 22))
        self.assertEqual(h.shape, (18, 22))
        self.assertEqual(d.shape, (18, 22))
        self.assertAlmostEqual(np.float(np.std(a)), 615.3699951171875, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 93.22431945800781, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 336.55194091796875, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 111.64820861816406, places=5)

        a, v, h, d = adm_dwt2_cy(a)
        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))
        self.assertAlmostEqual(np.float(np.std(a)), 814.2235717773438, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 152.11126708984375, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 451.4018249511719, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 209.05331420898438, places=5)


class AdmDwt2CyTestOnAkiyoXsmall(unittest.TestCase):

    def setUp(self) -> None:
        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "refp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_18x22"),
                width=22, height=18, yuv_type='yuv420p'
        ) as yuv_reader_ref:
            self.y_ref = yuv_reader_ref.next()[0].astype(np.float32)
        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "disp_vmaf_hacking_investigation_0_0_akiyo_cif_notyuv_0to0_identity_vs_akiyo_cif_notyuv_0to0_multiply_q_18x22"),
                width=22, height=18, yuv_type='yuv420p'
        ) as yuv_reader_ref:
            self.y_dis = yuv_reader_ref.next()[0].astype(np.float32)

    def test_adm_dwt2_cy_on_akiyo_single_scale(self):

        a, v, h, d = adm_dwt2_cy((self.y_ref - 128.0).astype(np.float32))
        self.assertEqual(a.shape, (9, 11))
        self.assertEqual(v.shape, (9, 11))
        self.assertEqual(h.shape, (9, 11))
        self.assertEqual(d.shape, (9, 11))
        self.assertAlmostEqual(np.float(np.std(a)), 59.79219055175781, places=5)
        self.assertAlmostEqual(np.float(np.std(v)), 12.54729175567627, places=5)
        self.assertAlmostEqual(np.float(np.std(h)), 35.61844253540039, places=5)
        self.assertAlmostEqual(np.float(np.std(d)), 15.827298164367676, places=5)
        self.assertAlmostEqual(np.float(np.mean(a)), -75.7516098022461, places=5)
        self.assertAlmostEqual(np.float(np.mean(v)), -2.64847469329834, places=5)
        self.assertAlmostEqual(np.float(np.mean(h)), -0.5888747572898865, places=5)
        self.assertAlmostEqual(np.float(np.mean(d)), -0.11442910879850388, places=5)

        self.assertAlmostEqual(a[0][0], -145.910904, places=6)
        self.assertAlmostEqual(a[0][-1], -140.545441, places=6)
        self.assertAlmostEqual(a[1][0], 0.0, places=6)
        self.assertAlmostEqual(a[1][-1], -113.189041, places=6)
        self.assertAlmostEqual(a[-1][0], -138.68803, places=5)
        self.assertAlmostEqual(a[-1][-1], -166.480515, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
