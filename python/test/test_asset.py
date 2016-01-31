__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import unittest

from asset import Asset

class AssetTest(unittest.TestCase):

    def test_workdir(self):
        import re
        asset = Asset(dataset="test", ref_path="", dis_path="", asset_dict={},
                      workdir_root="my_workdir_root")
        workdir = asset.workdir
        mo = re.match(r"my_workdir_root/[a-zA-Z0-9]+", workdir)
        self.assertTrue(mo is not None)

    def test_ref_width_height(self):
        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'ref_width':1920, 'ref_height':1080,})
        self.assertEquals(asset.ref_width_height, (1920, 1080))

        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'ref_width':1920, 'ref_height':1080,
                                  'width':720, 'height':480})
        self.assertEquals(asset.ref_width_height, (1920, 1080))

        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict= {'width':720, 'height':480})
        self.assertEquals(asset.ref_width_height, (720, 480))

    def test_dis_width_height(self):
        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'dis_width':1920, 'dis_height':1080,})
        self.assertEquals(asset.dis_width_height, (1920, 1080))

        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'dis_width':1920, 'dis_height':1080,
                                  'width':720, 'height':480})
        self.assertEquals(asset.dis_width_height, (1920, 1080))

        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict= {'width':720, 'height':480})
        self.assertEquals(asset.dis_width_height, (720, 480))

    def test_quality_width_height(self):
        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={"ref_width":1920, "ref_height":1080,
                                  "dis_width":720, "dis_height":480},)
        with self.assertRaises(AssertionError):
            print asset.quality_width_height

        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={"ref_width":1920, "ref_height":1080,
                                  "dis_width":720, "dis_height":480,
                                  "quality_width":1280, "quality_height":720},)
        self.assertEquals(asset.quality_width_height, (1280, 720))

    def test_start_end_frame(self):
        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'ref_start_frame':2, 'ref_end_frame':2,
                                  'dis_start_frame':3, 'dis_end_frame':3},)
        self.assertEquals(asset.ref_start_end_frame, (2, 2))
        self.assertEquals(asset.dis_start_end_frame, (3, 3))

        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'start_frame':2, 'end_frame':2})
        self.assertEquals(asset.ref_start_end_frame, (2, 2))
        self.assertEquals(asset.dis_start_end_frame, (2, 2))

        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'fps':24, 'duration_sec':2})
        self.assertEquals(asset.ref_start_end_frame, (0, 47))
        self.assertEquals(asset.dis_start_end_frame, (0, 47))

        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'fps':24, 'start_sec':2, 'end_sec': 3})
        self.assertEquals(asset.ref_start_end_frame, (48, 71))
        self.assertEquals(asset.dis_start_end_frame, (48, 71))

    def test_duration_sec(self):
        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'ref_start_frame':2, 'ref_end_frame':2,
                                  'dis_start_frame':3, 'dis_end_frame':3},)
        self.assertEquals(asset.ref_duration_sec, None)
        self.assertEquals(asset.dis_duration_sec, None)

        asset = Asset(dataset="test", ref_path="", dis_path="",
                      asset_dict={'ref_start_frame':0, 'ref_end_frame':23,
                                  'dis_start_frame':3, 'dis_end_frame':26,
                                  'fps':24},)
        self.assertEquals(asset.ref_duration_sec, 1.0)
        self.assertEquals(asset.dis_duration_sec, 1.0)

    def test_bitrate(self):
        ref_path = "../../resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = "../../resource/yuv/src01_hrc01_576x324.yuv"

        asset = Asset(dataset="test", ref_path=ref_path, dis_path=dis_path,
                      asset_dict={'ref_start_frame':0, 'ref_end_frame':47,
                                  'dis_start_frame':0, 'dis_end_frame':47,
                                  'fps':23.976},)
        self.assertEquals(asset.ref_bitrate_kbps_for_entire_file,
                          53693.964287999996)
        self.assertEquals(asset.dis_bitrate_kbps_for_entire_file,
                          53693.964287999996)

    def test_str(self):
        asset = Asset(dataset="test", ref_path="dir/refvideo.yuv",
                      dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})
        expected_str = "test_refvideo_720x480_2to2_vs_disvideo_720x480_2to2"
        self.assertEquals(str(asset), expected_str)

        asset = Asset(dataset="test", ref_path="dir/refvideo.yuv",
                      dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,})
        expected_str = "test_refvideo_720x480_vs_disvideo_720x480"
        self.assertEquals(str(asset), expected_str)

        asset = Asset(dataset="test", ref_path="dir/refvideo.yuv",
                      dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':1920, 'quality_height':1080})
        expected_str = "test_refvideo_720x480_vs_disvideo_720x480_q_1920x1080"
        self.assertEquals(str(asset), expected_str)

if __name__ == '__main__':
    unittest.main()
