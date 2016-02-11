from asset import Asset
import config
from result_store import ResultStore
from vmaf_quality_runner import VmafQualityRunner

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

class QualityResultTest(unittest.TestCase):

    def setUp(self):
        ref_path = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv"
        dis_path = config.ROOT + "/resource/yuv/checkerboard_1920_1080_10_3_1_0.yuv"
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      workdir_root=config.ROOT + "/workspace/workdir",
                      ref_path=ref_path,
                      dis_path=dis_path,
                      asset_dict={'width':1920, 'height':1080})

        self.runner = VmafQualityRunner(
            [asset], None, fifo_mode=True,
            log_file_dir=config.ROOT + "/workspace/log_file_dir")
        self.runner.run()
        self.result = self.runner.results[0]

    def tearDown(self):
        if hasattr(self, 'runner'): self.runner.remove_logs()
        pass

    @unittest.skip("Skip dataframe test for now...")
    def test_to_dataframe(self):
        print 'test on quality result to dataframe...'
        df = self.result.to_dataframe()
        self.assertEquals(len(df), 15)
        self.assertEquals(df.loc[df['score_key'] == 'VMAF_score'] \
                          ['score'].mean(), 43.46099858503333)
        self.assertEquals(df.loc[df['score_key'] == 'VMAF_adm_score'] \
                              ['score'].mean(), 0.81386)
        self.assertEquals(df.loc[df['score_key'] == 'VMAF_vif_score'] \
                              ['score'].mean(), 0.15612933333333334)
        self.assertEquals(df.loc[df['score_key'] == 'VMAF_motion_score'] \
                              ['score'].mean(), 12.343795333333333)
        self.assertEquals(df.loc[df['score_key'] == 'VMAF_ansnr_score'] \
                              ['score'].mean(), 12.418291000000002)

    def test_to_score_str(self):
        print 'test on quality result aggregate scores...'
        self.assertEquals(self.result.get_score('VMAF_score'),
                          43.46099858503333)
        self.assertEquals(self.result.get_score('VMAF_adm_score'),
                          0.81386)
        self.assertEquals(self.result.get_score('VMAF_vif_score'),
                          0.15612933333333334)
        self.assertEquals(self.result.get_score('VMAF_motion_score'),
                          12.343795333333333)
        self.assertEquals(self.result.get_score('VMAF_ansnr_score'),
                          12.418291000000002)

        with self.assertRaises(KeyError):
            self.result.get_score('VVMAF_score')
        with self.assertRaises(KeyError):
            self.result.get_score('VMAF_motion_scor')

        self.assertEquals(
            self.result._get_aggregate_score_str(),
            "Aggregate: VMAF_adm_score:0.814, VMAF_vif_score:0.156, "
            "VMAF_motion_score:12.344, VMAF_score:43.461, "
            "VMAF_ansnr_score:12.418")

    def test_from_dataframe(self):
        df = self.result.to_dataframe()
        result_recon = ResultStore.from_dataframe(df)
