import unittest

from test.testutil import set_default_576_324_videos_for_testing
from vmaf.config import VmafConfig
from vmaf.core.quality_runner import PsnrQualityRunner
from vmaf.tools.misc import MyTestCase, QualityRunnerTestMixin, import_python_file


class QualityRunnerTestMixinTest(MyTestCase, QualityRunnerTestMixin):

    def setUp(self):
        super().setUp()
        ref_path, dis_path, asset, asset_original = set_default_576_324_videos_for_testing()
        self.data = (
            [30.755063979166664, PsnrQualityRunner, asset, None],
        )

    def test_run_each(self):
        for data_each in self.data:
            self.run_each(*data_each)

    def test_plot_frame_scores(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for data_each in self.data:
            new_data_each = data_each[1:]
            self.plot_frame_scores(ax, *new_data_each, label='src01_hrc01_576x324.yuv')
        ax.legend()
        ax.grid()
        # from vmaf.config import DisplayConfig; DisplayConfig.show(write_to_dir=None)


class MiscTest(unittest.TestCase):

    def test_import_python_file(self):
        ds = import_python_file(VmafConfig.test_resource_path('example_raw_dataset2.py'))
        self.assertEqual(ds.dataset_name, 'example')
        self.assertEqual(ds.yuv_fmt, 'yuv420p')
        self.assertEqual(ds.ref_videos[0]['content_id'], 0)
        self.assertEqual(ds.ref_videos[0]['path'], '/dir/yuv/checkerboard_1920_1080_10_3_0_0.yuv')
        self.assertEqual(ds.dis_videos[3]['content_id'], 1)
        self.assertEqual(ds.dis_videos[3]['asset_id'], 3)
        self.assertEqual(ds.dis_videos[3]['os'], [70, 75, 80, 85, 90])
        self.assertEqual(ds.dis_videos[3]['path'], '/dir/yuv/flat_1920_1080_10.yuv')

    def test_import_python_file_with_override(self):
        ds = import_python_file(VmafConfig.test_resource_path('example_raw_dataset2.py'), override={'dir': '/xyz'})
        self.assertEqual(ds.dataset_name, 'example')
        self.assertEqual(ds.yuv_fmt, 'yuv420p')
        self.assertEqual(ds.ref_videos[0]['content_id'], 0)
        self.assertEqual(ds.ref_videos[0]['path'], '/xyz/yuv/checkerboard_1920_1080_10_3_0_0.yuv')
        self.assertEqual(ds.dis_videos[3]['content_id'], 1)
        self.assertEqual(ds.dis_videos[3]['asset_id'], 3)
        self.assertEqual(ds.dis_videos[3]['os'], [70, 75, 80, 85, 90])
        self.assertEqual(ds.dis_videos[3]['path'], '/xyz/yuv/flat_1920_1080_10.yuv')

    def test_import_python_file_with_override_and_add_new(self):
        ds = import_python_file(VmafConfig.test_resource_path('example_raw_dataset3.py'),
                                override={'dir': '/xyz', 'quality_width': 960, 'quality_height': 540})
        self.assertEqual(ds.dataset_name, 'example')
        self.assertEqual(ds.yuv_fmt, 'yuv420p')
        self.assertEqual(ds.ref_videos[0]['content_id'], 0)
        self.assertEqual(ds.ref_videos[0]['path'], '/xyz/yuv/checkerboard_1920_1080_10_3_0_0.yuv')
        self.assertEqual(ds.dis_videos[3]['content_id'], 1)
        self.assertEqual(ds.dis_videos[3]['asset_id'], 3)
        self.assertEqual(ds.dis_videos[3]['os'], [70, 75, 80, 85, 90])
        self.assertEqual(ds.dis_videos[3]['path'], '/xyz/yuv/flat_1920_1080_10.yuv')
        self.assertEqual(ds.quality_width, 960)
        self.assertEqual(ds.quality_height, 540)

    def test_import_python_file_with_override_multiple_in_one_line(self):
        ds = import_python_file(VmafConfig.test_resource_path('example_raw_dataset4.py'),
                                override={'dir': '/xyz', 'quality_width': 960, 'quality_height': 540})
        self.assertEqual(ds.dataset_name, 'example')
        self.assertEqual(ds.yuv_fmt, 'yuv420p')
        self.assertEqual(ds.ref_videos[0]['content_id'], 0)
        self.assertEqual(ds.ref_videos[0]['path'], '/xyz/yuv/checkerboard_1920_1080_10_3_0_0.yuv')
        self.assertEqual(ds.dis_videos[3]['content_id'], 1)
        self.assertEqual(ds.dis_videos[3]['asset_id'], 3)
        self.assertEqual(ds.dis_videos[3]['os'], [70, 75, 80, 85, 90])
        self.assertEqual(ds.dis_videos[3]['path'], '/xyz/yuv/flat_1920_1080_10.yuv')
        self.assertEqual(ds.quality_width, 960)
        self.assertEqual(ds.quality_height, 540)

