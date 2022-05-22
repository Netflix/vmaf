from test.testutil import set_default_576_324_videos_for_testing
from vmaf.core.quality_runner import PsnrQualityRunner
from vmaf.tools.misc import MyTestCase, QualityRunnerTestMixin


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
