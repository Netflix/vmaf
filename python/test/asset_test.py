__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest

import config
from asset import Asset

class AssetTest(unittest.TestCase):

    def test_workdir(self):
        import re
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={}, workdir_root="my_workdir_root")
        workdir = asset.workdir
        self.assertTrue(re.match(r"^my_workdir_root/[a-zA-Z0-9-]+$", workdir))

    def test_ref_width_height(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_width':1920, 'ref_height':1080,})
        self.assertEquals(asset.ref_width_height, (1920, 1080))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_width':1920, 'ref_height':1080,
                                  'width':720, 'height':480})
        self.assertEquals(asset.ref_width_height, (1920, 1080))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict= {'width':720, 'height':480})
        self.assertEquals(asset.ref_width_height, (720, 480))

    def test_dis_width_height(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_width':1920, 'dis_height':1080,})
        self.assertEquals(asset.dis_width_height, (1920, 1080))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_width':1920, 'dis_height':1080,
                                  'width':720, 'height':480})
        self.assertEquals(asset.dis_width_height, (1920, 1080))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict= {'width':720, 'height':480})
        self.assertEquals(asset.dis_width_height, (720, 480))

    def test_quality_width_height(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={"ref_width":1920, "ref_height":1080,
                                  "dis_width":720, "dis_height":480},)
        with self.assertRaises(AssertionError):
            print asset.quality_width_height

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={"ref_width":1920, "ref_height":1080,
                                  "dis_width":720, "dis_height":480,
                                  "quality_width":1280, "quality_height":720},)
        self.assertEquals(asset.quality_width_height, (1280, 720))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={"ref_width":720, "ref_height":480,
                                  "dis_width":720, "dis_height":480,},)
        self.assertEquals(asset.quality_width_height, (720, 480))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={"width":720, "height":480,},)
        self.assertEquals(asset.quality_width_height, (720, 480))

    def test_start_end_frame(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_start_frame':2, 'ref_end_frame':2,
                                  'dis_start_frame':3, 'dis_end_frame':3},)
        self.assertEquals(asset.ref_start_end_frame, (2, 2))
        self.assertEquals(asset.dis_start_end_frame, (3, 3))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'start_frame':2, 'end_frame':2})
        self.assertEquals(asset.ref_start_end_frame, (2, 2))
        self.assertEquals(asset.dis_start_end_frame, (2, 2))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps':24, 'duration_sec':2})
        self.assertEquals(asset.ref_start_end_frame, (0, 47))
        self.assertEquals(asset.dis_start_end_frame, (0, 47))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps':24, 'start_sec':2, 'end_sec': 3})
        self.assertEquals(asset.ref_start_end_frame, (48, 71))
        self.assertEquals(asset.dis_start_end_frame, (48, 71))

    def test_duration_sec(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_start_frame':2, 'ref_end_frame':2,
                                  'dis_start_frame':3, 'dis_end_frame':3},)
        self.assertEquals(asset.ref_duration_sec, None)
        self.assertEquals(asset.dis_duration_sec, None)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_start_frame':0, 'ref_end_frame':23,
                                  'dis_start_frame':3, 'dis_end_frame':26,
                                  'fps':24},)
        self.assertEquals(asset.ref_duration_sec, 1.0)
        self.assertEquals(asset.dis_duration_sec, 1.0)

    def test_bitrate(self):
        ref_path = config.ROOT + "/resource/yuv/src01_hrc00_576x324.yuv"
        dis_path = config.ROOT + "/resource/yuv/src01_hrc01_576x324.yuv"

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path=ref_path, dis_path=dis_path,
                      asset_dict={'ref_start_frame':0, 'ref_end_frame':47,
                                  'dis_start_frame':0, 'dis_end_frame':47,
                                  'fps':23.976},)
        self.assertEquals(asset.ref_bitrate_kbps_for_entire_file,
                          53693.964287999996)
        self.assertEquals(asset.dis_bitrate_kbps_for_entire_file,
                          53693.964287999996)

    def test_ref_dis_str(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})
        self.assertEquals(
            asset.to_string(),
            "test_0_0_refvideo_720x480_2to2_vs_disvideo_720x480_2to2_q_720x480"
        )
        self.assertEquals(
            asset.to_string_compact(),
            "test_0_0_disvideo_720x480_2to2_q_720x480"
        )
        self.assertEquals(
            str(asset),
            "test_0_0_disvideo_720x480_2to2_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,})
        self.assertEquals(
            asset.to_string(),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480"
        )
        self.assertEquals(
            asset.to_string_compact(),
            "test_0_1_disvideo_720x480_q_720x480"
        )
        self.assertEquals(
            str(asset),
            "test_0_1_disvideo_720x480_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':1920, 'quality_height':1080})
        self.assertEquals(
            asset.to_string(),
            "test_0_2_refvideo_720x480_vs_disvideo_720x480_q_1920x1080"
        )
        self.assertEquals(
            asset.to_string_compact(),
            "test_0_2_disvideo_720x480_q_1920x1080"
        )
        self.assertEquals(
            str(asset),
            "test_0_2_disvideo_720x480_q_1920x1080"
        )

    def test_workfile_path(self):
        import re
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2,
                                  'quality_width':1920, 'quality_height':1080},
                      workdir_root="workdir")
        expected_ref_workfile_path_re = \
            r"^workdir/[a-zA-Z0-9-]+/ref_test_0_0_disvideo_720x480_2to2_q_1920x1080$"
        expected_dis_workfile_path_re = \
            r"^workdir/[a-zA-Z0-9-]+/dis_test_0_0_disvideo_720x480_2to2_q_1920x1080"
        self.assertTrue(re.match(expected_ref_workfile_path_re, asset.ref_workfile_path))
        self.assertTrue(re.match(expected_dis_workfile_path_re, asset.dis_workfile_path))

    def test_yuv_type(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps':24, 'start_sec':2, 'end_sec': 3})
        self.assertEquals(asset.yuv_type, 'yuv420p')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="", asset_dict={
            'fps':24, 'start_sec':2, 'end_sec': 3, 'yuv_type':'yuv444p'})
        self.assertEquals(asset.yuv_type, 'yuv444p')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="", asset_dict={
            'fps':24, 'start_sec':2, 'end_sec': 3, 'yuv_type':'yuv444a'})
        with self.assertRaises(AssertionError):
            print asset.yuv_type


if __name__ == '__main__':
    unittest.main()
