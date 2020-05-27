import unittest

import numpy as np

from vmaf.core.adm_dwt2_cy import adm_dwt2_cy
from vmaf.core.adm_dwt2_py import adm_dwt2_py


class AdmDwt2CyTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_adm_dwt2_cy(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[324, 576])
        a, ds = adm_dwt2_cy(x.astype(np.float32))
        self.assertEqual(a.shape, (162, 288))
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0].shape, (162, 288))
        self.assertEqual(ds[1].shape, (162, 288))
        self.assertEqual(ds[2].shape, (162, 288))
        self.assertAlmostEqual(np.float(np.std(a)), 73.94279479980469, places=6)
        self.assertAlmostEqual(np.float(np.std(ds[0])), 73.30350494384766, places=6)
        self.assertAlmostEqual(np.float(np.std(ds[1])), 73.6191713793981, places=6)
        self.assertAlmostEqual(np.float(np.std(ds[2])), 73.19024658203125, places=6)

    def test_adm_dwt2_py(self):
        np.random.seed(0)
        x = np.random.uniform(low=-128, high=127, size=[324, 576])
        a, ds = adm_dwt2_py(x)
        self.assertEqual(a.shape, (162, 288))
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0].shape, (162, 288))
        self.assertEqual(ds[1].shape, (162, 288))
        self.assertEqual(ds[2].shape, (162, 288))
        self.assertAlmostEqual(np.float(np.std(a)), 73.8959273922819, places=6)
        self.assertAlmostEqual(np.float(np.std(ds[0])), 73.5355886699825, places=6)
        self.assertAlmostEqual(np.float(np.std(ds[1])), 73.69196438463929, places=6)
        self.assertAlmostEqual(np.float(np.std(ds[2])), 73.52173319007242, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
