__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import unittest

import numpy as np

from vmaf.config import VmafConfig
from vmaf.tools.reader import YuvReader


class YuvReaderTest(unittest.TestCase):

    def test_yuv_reader(self):
        with YuvReader(
            filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            width=576,
            height=324,
            yuv_type='yuv420p'
        ) as yuv_reader:
            self.assertEqual(yuv_reader.num_bytes, 13436928)
            self.assertEqual(yuv_reader.num_frms, 48)
            self.assertEqual(yuv_reader._get_uv_width_height_multiplier(), (0.5, 0.5))

    def test_with(self):
        with YuvReader(
            filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            width=576,
            height=324,
            yuv_type='yuv420p'
        ) as yuv_reader:
            assert hasattr(yuv_reader.file, "read")

    def test_next_y_u_v(self):
        with YuvReader(
            filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            width=576,
            height=324,
            yuv_type='yuv420p'
        ) as yuv_reader:

            y, u, v = yuv_reader.next()
            y, u, v = y.astype(np.double), u.astype(np.double), v.astype(np.double)

            self.assertEqual(y[0][0], 87)
            self.assertEqual(y[0][1], 131)
            self.assertEqual(y[1][0], 95)

            self.assertEqual(u[0][0], 92)
            self.assertEqual(u[0][1], 97)
            self.assertEqual(u[1][0], 90)

            self.assertEqual(v[0][0], 121)
            self.assertEqual(v[0][1], 126)
            self.assertEqual(v[1][0], 122)

            self.assertAlmostEqual(y.mean(), 61.928749785665296, places=4)
            self.assertAlmostEqual(u.mean(), 114.6326517489712, places=4)
            self.assertAlmostEqual(v.mean(), 122.05084019204389, places=4)

            y, u, v = yuv_reader.next()
            y, u, v = y.astype(np.double), u.astype(np.double), v.astype(np.double)

            self.assertEqual(y[0][0], 142)
            self.assertEqual(y[0][1], 128)
            self.assertEqual(y[1][0], 134)

            self.assertEqual(u[0][0], 93)
            self.assertEqual(u[0][1], 102)
            self.assertEqual(u[1][0], 91)

            self.assertEqual(v[0][0], 128)
            self.assertEqual(v[0][1], 126)
            self.assertEqual(v[1][0], 124)

            self.assertAlmostEqual(y.mean(), 61.265260631001375, places=4)
            self.assertAlmostEqual(u.mean(), 114.72515860768175, places=4)
            self.assertAlmostEqual(v.mean(), 122.12022033607681, places=4)

    def test_next_y_u_v_gray(self):

        with YuvReader(
            filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv"),
            width=576,
            height=324,
            yuv_type='gray'
        ) as yuv_reader:

            y, u, v = yuv_reader.next()
            y = y.astype(np.double)

            self.assertEqual(y[0][0], 87)
            self.assertTrue(u is None)
            self.assertTrue(v is None)

            self.assertAlmostEqual(y.mean(), 61.928749785665296, places=4)

            y, u, v = yuv_reader.next()
            y = y.astype(np.double)

            self.assertEqual(y[0][0], 92)
            self.assertTrue(u is None)
            self.assertTrue(v is None)

    def test_iteration(self):

        y_1stmoments = []
        y_2ndmoments = []

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"),
                width=576, height=324, yuv_type='yuv420p') as yuv_reader:

            for y, u, v in yuv_reader:
                y, u, v = y.astype(np.double), u.astype(np.double), v.astype(np.double)
                y_1stmoments.append(y.mean())
                y_2ndmoments.append(y.var() + y.mean() * y.mean())

        self.assertEqual(len(y_1stmoments), 48)
        self.assertEqual(len(y_2ndmoments), 48)
        self.assertAlmostEqual(float(np.mean(y_1stmoments)), 61.332006624999984, places=4)
        self.assertAlmostEqual(float(np.mean(y_2ndmoments)), 4798.659574041666, places=4)

    def test_iteration_manual(self):

        y_1stmoments = []
        y_2ndmoments = []

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"),
                width=576, height=324, yuv_type='yuv420p') as yuv_reader:

            while True:
                try:
                    y, u, v = yuv_reader.next()
                    y, u, v = y.astype(np.double), u.astype(np.double), v.astype(np.double)
                    y_1stmoments.append(y.mean())
                    y_2ndmoments.append(y.var() + y.mean() * y.mean())
                except StopIteration:
                    break

        self.assertEqual(len(y_1stmoments), 48)
        self.assertEqual(len(y_2ndmoments), 48)
        self.assertAlmostEqual(float(np.mean(y_1stmoments)), 61.332006624999984, places=4)
        self.assertAlmostEqual(float(np.mean(y_2ndmoments)), 4798.659574041666, places=4)

    def test_iteration_manual_float(self):

        y_1stmoments = []
        y_2ndmoments = []

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv"),
                width=576, height=324, yuv_type='yuv420p') as yuv_reader:

            while True:
                try:
                    y, u, v = yuv_reader.next(format='float')
                    y_1stmoments.append(y.mean())
                    y_2ndmoments.append(y.var() + y.mean() * y.mean())
                except StopIteration:
                    break

        self.assertEqual(len(y_1stmoments), 48)
        self.assertEqual(len(y_2ndmoments), 48)
        self.assertAlmostEqual(float(np.mean(y_1stmoments)), 0.24051767285953876, places=4)
        self.assertAlmostEqual(float(np.mean(y_2ndmoments)), 0.07379714839039883, places=4)


class YuvReaderTest10le(unittest.TestCase):

    def test_yuv_reader(self):

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv422p10le.yuv"),
                width=576,
                height=324,
                yuv_type='yuv422p10le'
        ) as yuv_reader:
            self.assertEqual(yuv_reader.num_bytes, 35831808)
            self.assertEqual(yuv_reader.num_frms, 48)
            self.assertEqual(yuv_reader._get_uv_width_height_multiplier(), (0.5, 1.0))

    def test_with(self):

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv422p10le.yuv"),
                width=576,
                height=324,
                yuv_type='yuv422p10le'
        ) as yuv_reader:

            y, u, v = yuv_reader.next()
            y, u, v = y.astype(np.double) / 4.0, u.astype(np.double) / 4.0, v.astype(np.double) / 4.0

            self.assertEqual(y[0][0], 87)
            self.assertEqual(y[0][1], 131)
            self.assertEqual(y[1][0], 95)

            self.assertEqual(u[0][0], 92.25)
            self.assertEqual(u[0][1], 97.5)
            self.assertEqual(u[1][0], 91.75)

            self.assertEqual(v[0][0], 121)
            self.assertEqual(v[0][1], 126.25)
            self.assertEqual(v[1][0], 121.25)

            self.assertAlmostEqual(y.mean(), 61.928749785665296, places=4)
            self.assertAlmostEqual(u.mean(), 114.63283661265432, places=4)
            self.assertAlmostEqual(v.mean(), 122.05113490226337, places=4)

            y, u, v = yuv_reader.next()
            y, u, v = y.astype(np.double) / 4.0, u.astype(np.double) / 4.0, v.astype(np.double) / 4.0

            self.assertEqual(y[0][0], 142)
            self.assertEqual(y[0][1], 128)
            self.assertEqual(y[1][0], 134)

            self.assertEqual(u[0][0], 93.25)
            self.assertEqual(u[0][1], 102.75)
            self.assertEqual(u[1][0], 92.75)

            self.assertEqual(v[0][0], 128.25)
            self.assertEqual(v[0][1], 126.5)
            self.assertEqual(v[1][0], 127.25)

            self.assertAlmostEqual(y.mean(), 61.265260631001375, places=4)
            self.assertAlmostEqual(u.mean(), 114.72527917095336, places=4)
            self.assertAlmostEqual(v.mean(), 122.12047217935527, places=4)

    def test_iteration(self):

        y_1stmoments = []
        y_2ndmoments = []

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv422p10le.yuv"),
                width=576, height=324, yuv_type='yuv422p10le') as yuv_reader:

            for y, u, v in yuv_reader:
                y, u, v = y.astype(np.double) / 4.0, u.astype(np.double) / 4.0, v.astype(np.double) / 4.0
                y_1stmoments.append(y.mean())
                y_2ndmoments.append(y.var() + y.mean() * y.mean())

        self.assertEqual(len(y_1stmoments), 48)
        self.assertEqual(len(y_2ndmoments), 48)
        self.assertAlmostEqual(float(np.mean(y_1stmoments)), 61.332006624999984, places=4)
        self.assertAlmostEqual(float(np.mean(y_2ndmoments)), 4798.659574041666, places=4)


class YuvReaderTest12le(unittest.TestCase):

    def test_yuv_reader(self):

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv420p12le.yuv"),
                width=576,
                height=324,
                yuv_type='yuv420p12le'
        ) as yuv_reader:
            self.assertEqual(yuv_reader.num_bytes, 1679616)
            self.assertEqual(yuv_reader.num_frms, 3)
            self.assertEqual(yuv_reader._get_uv_width_height_multiplier(), (0.5, 0.5))

    def test_with(self):

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv420p12le.yuv"),
                width=576,
                height=324,
                yuv_type='yuv420p12le'
        ) as yuv_reader:

            y, u, v = yuv_reader.next()
            y, u, v = y.astype(np.double) / 16.0, u.astype(np.double) / 16.0, v.astype(np.double) / 16.0

            self.assertEqual(y[0][0], 87)
            self.assertEqual(y[0][1], 131)
            self.assertEqual(y[1][0], 95)

            self.assertEqual(u[0][0], 92)
            self.assertEqual(u[0][1], 97)
            self.assertEqual(u[1][0], 90)

            self.assertEqual(v[0][0], 121)
            self.assertEqual(v[0][1], 126)
            self.assertEqual(v[1][0], 122)

            self.assertAlmostEqual(y.mean(), 61.928749785665296, places=4)
            self.assertAlmostEqual(u.mean(), 114.6326517489712, places=4)
            self.assertAlmostEqual(v.mean(), 122.05084019204389, places=4)

            y, u, v = yuv_reader.next()
            y, u, v = y.astype(np.double) / 16.0, u.astype(np.double) / 16.0, v.astype(np.double) / 16.0

            self.assertEqual(y[0][0], 142)
            self.assertEqual(y[0][1], 128)
            self.assertEqual(y[1][0], 134)

            self.assertEqual(u[0][0], 93)
            self.assertEqual(u[0][1], 102)
            self.assertEqual(u[1][0], 91)

            self.assertEqual(v[0][0], 128)
            self.assertEqual(v[0][1], 126)
            self.assertEqual(v[1][0], 124)

            self.assertAlmostEqual(y.mean(), 61.265260631001375, places=4)
            self.assertAlmostEqual(u.mean(), 114.72515860768175, places=4)
            self.assertAlmostEqual(v.mean(), 122.12022033607681, places=4)

    def test_iteration(self):

        y_1stmoments = []
        y_2ndmoments = []

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv420p12le.yuv"),
                width=576, height=324, yuv_type='yuv420p12le') as yuv_reader:

            for y, u, v in yuv_reader:
                y, u, v = y.astype(np.double) / 16.0, u.astype(np.double) / 16.0, v.astype(np.double) / 16.0
                y_1stmoments.append(y.mean())
                y_2ndmoments.append(y.var() + y.mean() * y.mean())

        self.assertEqual(len(y_1stmoments), 3)
        self.assertEqual(len(y_2ndmoments), 3)
        self.assertAlmostEqual(float(np.mean(y_1stmoments)), 62.267612954389584, places=4)
        self.assertAlmostEqual(float(np.mean(y_2ndmoments)), 4904.42749592764, places=4)


class YuvReaderTest16le(unittest.TestCase):

    def test_yuv_reader(self):

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv420p16le.yuv"),
                width=576,
                height=324,
                yuv_type='yuv420p16le'
        ) as yuv_reader:
            self.assertEqual(yuv_reader.num_bytes, 1679616)
            self.assertEqual(yuv_reader.num_frms, 3)
            self.assertEqual(yuv_reader._get_uv_width_height_multiplier(), (0.5, 0.5))

    def test_with(self):

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv420p16le.yuv"),
                width=576,
                height=324,
                yuv_type='yuv420p16le'
        ) as yuv_reader:

            y, u, v = yuv_reader.next()
            y, u, v = y.astype(np.double) / 256.0, u.astype(np.double) / 256.0, v.astype(np.double) / 256.0

            self.assertEqual(y[0][0], 87)
            self.assertEqual(y[0][1], 131)
            self.assertEqual(y[1][0], 95)

            self.assertEqual(u[0][0], 92)
            self.assertEqual(u[0][1], 97)
            self.assertEqual(u[1][0], 90)

            self.assertEqual(v[0][0], 121)
            self.assertEqual(v[0][1], 126)
            self.assertEqual(v[1][0], 122)

            self.assertAlmostEqual(y.mean(), 61.928749785665296, places=4)
            self.assertAlmostEqual(u.mean(), 114.6326517489712, places=4)
            self.assertAlmostEqual(v.mean(), 122.05084019204389, places=4)

            y, u, v = yuv_reader.next()
            y, u, v = y.astype(np.double) / 256.0, u.astype(np.double) / 256.0, v.astype(np.double) / 256.0

            self.assertEqual(y[0][0], 142)
            self.assertEqual(y[0][1], 128)
            self.assertEqual(y[1][0], 134)

            self.assertEqual(u[0][0], 93)
            self.assertEqual(u[0][1], 102)
            self.assertEqual(u[1][0], 91)

            self.assertEqual(v[0][0], 128)
            self.assertEqual(v[0][1], 126)
            self.assertEqual(v[1][0], 124)

            self.assertAlmostEqual(y.mean(), 61.265260631001375, places=4)
            self.assertAlmostEqual(u.mean(), 114.72515860768175, places=4)
            self.assertAlmostEqual(v.mean(), 122.12022033607681, places=4)

    def test_iteration(self):

        y_1stmoments = []
        y_2ndmoments = []

        with YuvReader(
                filepath=VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv420p16le.yuv"),
                width=576, height=324, yuv_type='yuv420p16le') as yuv_reader:

            for y, u, v in yuv_reader:
                y, u, v = y.astype(np.double) / 256.0, u.astype(np.double) / 256.0, v.astype(np.double) / 256.0
                y_1stmoments.append(y.mean())
                y_2ndmoments.append(y.var() + y.mean() * y.mean())

        self.assertEqual(len(y_1stmoments), 3)
        self.assertEqual(len(y_2ndmoments), 3)
        self.assertAlmostEqual(float(np.mean(y_1stmoments)), 62.267612954389584, places=4)
        self.assertAlmostEqual(float(np.mean(y_2ndmoments)), 4904.42749592764, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
