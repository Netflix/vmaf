__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import unittest
import re

import config
from core.asset import Asset


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
        self.assertAlmostEquals(asset.ref_bitrate_kbps_for_entire_file,
                          53693.964287999996, places=4)
        self.assertAlmostEquals(asset.dis_bitrate_kbps_for_entire_file,
                          53693.964287999996, places=4)

    def test_to_normalized_dict(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})
        self.assertEquals(
            asset.to_normalized_dict(),
            {'asset_dict': {'end_frame': 2, 'height': 480,
                            'start_frame': 2, 'width': 720},
              'asset_id': 0,
              'content_id': 0,
              'dataset': 'test',
              'dis_path': 'disvideo.yuv',
              'ref_path': 'refvideo.yuv',
              'workdir': ''
             }
        )

    def test_to_normalized_dict_10le(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv420p10le.yuv",
                      dis_path="dir/disvideo.yuv420p10le.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})
        self.assertEquals(
            asset.to_normalized_dict(),
            {'asset_dict': {'end_frame': 2, 'height': 480,
                            'start_frame': 2, 'width': 720},
              'asset_id': 0,
              'content_id': 0,
              'dataset': 'test',
              'dis_path': 'disvideo.yuv420p10le.yuv',
              'ref_path': 'refvideo.yuv420p10le.yuv',
              'workdir': ''
             }
        )

    def test_str_repr(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2})
        self.assertEquals(
            str(asset),
            "test_0_0_refvideo_720x480_2to2_vs_disvideo_720x480_2to2_q_720x480"
        )
        expected_repr = '{"asset_dict": {"end_frame": 2, "height": 480, "start_frame": 2, "width": 720}, "asset_id": 0, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}'
        self.assertEquals(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEquals(asset, recon_asset)
        self.assertTrue(asset == recon_asset)
        self.assertFalse(asset != recon_asset)

        self.assertEquals(asset.to_normalized_repr(), expected_repr)

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,})
        self.assertEquals(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480"
        )
        expected_repr = '{"asset_dict": {"height": 480, "width": 720}, "asset_id": 1, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}'
        self.assertEquals(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEquals(asset, recon_asset)

        asset = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':1920, 'quality_height':1080})
        self.assertEquals(
            str(asset),
            "test_0_2_refvideo_720x480_vs_disvideo_720x480_q_1920x1080"
        )
        expected_repr = '{"asset_dict": {"height": 480, "quality_height": 1080, "quality_width": 1920, "width": 720}, "asset_id": 2, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}'
        self.assertEquals(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEquals(asset, recon_asset)

        asset = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':1920, 'quality_height':1080,
                                  'yuv_type':'yuv422p'})
        self.assertEquals(
            str(asset),
            "test_0_2_refvideo_720x480_yuv422p_vs_disvideo_720x480_yuv422p_q_1920x1080"
        )
        expected_repr = '{"asset_dict": {"height": 480, "quality_height": 1080, "quality_width": 1920, "width": 720, "yuv_type": "yuv422p"}, "asset_id": 2, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}'
        self.assertEquals(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEquals(asset, recon_asset)

        asset = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':1920, 'quality_height':1080,
                                  'resampling_type':'lanczos'})
        self.assertEquals(
            str(asset),
            "test_0_2_refvideo_720x480_vs_disvideo_720x480_q_1920x1080_lanczos"
        )
        expected_repr = '{"asset_dict": {"height": 480, "quality_height": 1080, "quality_width": 1920, "resampling_type": "lanczos", "width": 720}, "asset_id": 2, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}'
        self.assertEquals(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEquals(asset, recon_asset)

    def test_str(self):

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':720, 'quality_height':480})
        self.assertEquals(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'resampling_type':'lanczos'})
        self.assertEquals(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':720, 'quality_height':480,
                                  'resampling_type':'lanczos'})
        self.assertEquals(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'ref_width':720, 'ref_height':480,
                                  'dis_width':1920, 'dis_height':1080,
                                  'quality_width':720, 'quality_height':480,
                                  'resampling_type':'lanczos'})
        self.assertEquals(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_1920x1080_q_720x480_lanczos"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'ref_width':1920, 'ref_height':1080,
                                  'dis_width':720, 'dis_height':480,
                                  'quality_width':720, 'quality_height':480,
                                  'resampling_type':'lanczos'})
        self.assertEquals(
            str(asset),
            "test_0_1_refvideo_1920x1080_vs_disvideo_720x480_q_720x480_lanczos"
        )

    def test_hash_equal(self):
        asset1 = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':1920, 'quality_height':1080})
        asset2 = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':1920, 'quality_height':1080})
        asset3 = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="my/dir/refvideo.yuv", dis_path="my/dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':1920, 'quality_height':1080})
        asset4 = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="my/dir/refvideo.yuv", dis_path="my/dir/disvideo.avi",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':1920, 'quality_height':1080})

        self.assertTrue(asset1 == asset2)
        self.assertTrue(asset2 == asset3)
        self.assertFalse(asset3 == asset4)
        self.assertTrue(hash(asset2) == hash(asset3))
        self.assertFalse(hash(asset1) == hash(asset4))

    def test_workfile_path(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2,
                                  'quality_width':1920, 'quality_height':1080},
                      workdir_root="workdir")
        expected_ref_workfile_path_re = \
            r"^workdir/[a-zA-Z0-9-]+/" \
            r"ref_test_0_0_refvideo_720x480_2to2_vs_disvideo_720x480_2to2_q_1920x1080"
        expected_dis_workfile_path_re = \
            r"^workdir/[a-zA-Z0-9-]+/" \
            r"dis_test_0_0_refvideo_720x480_2to2_vs_disvideo_720x480_2to2_q_1920x1080"
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

    def test_resampling_type(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps':24, 'start_sec':2, 'end_sec': 3})
        self.assertEquals(asset.resampling_type, 'bilinear')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps':24, 'start_sec':2, 'end_sec': 3,
                                  'resampling_type':'lanczos'})
        self.assertEquals(asset.resampling_type, 'lanczos')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps':24, 'start_sec':2, 'end_sec': 3,
                                  'resampling_type':'bicubic'})
        self.assertEquals(asset.resampling_type, 'bicubic')

    def test_use_path_as_workpath(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width':720, 'height':480,
                                  'start_frame':2, 'end_frame':2,
                                  'quality_width':1920, 'quality_height':1080},
                      workdir_root="workdir")
        expected_ref_workfile_path_re = \
            r"^workdir/[a-zA-Z0-9-]+/" \
            r"ref_test_0_0_refvideo_720x480_2to2_vs_disvideo_720x480_2to2_q_1920x1080"
        expected_dis_workfile_path_re = \
            r"^workdir/[a-zA-Z0-9-]+/" \
            r"dis_test_0_0_refvideo_720x480_2to2_vs_disvideo_720x480_2to2_q_1920x1080"
        self.assertTrue(re.match(expected_ref_workfile_path_re, asset.ref_workfile_path))
        self.assertTrue(re.match(expected_dis_workfile_path_re, asset.dis_workfile_path))
        self.assertFalse('use_path_as_workpath' in asset.asset_dict)

        asset.use_path_as_workpath = True
        self.assertTrue('use_path_as_workpath' in asset.asset_dict)
        self.assertTrue(asset.asset_dict['use_path_as_workpath'])
        self.assertEquals(asset.ref_workfile_path, 'dir/refvideo.yuv')
        self.assertEquals(asset.dis_workfile_path, 'dir/disvideo.yuv')

    def test_crop_cmd(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':720, 'quality_height':320,
                                  'yuv_type':'yuv422p',
                                  'crop_cmd':'570:320:3:2'})
        self.assertEquals(asset.crop_cmd, '570:320:3:2')
        self.assertEquals(str(asset), "test_0_0__720x480_yuv422p_vs__720x480_yuv422p_q_720x320_crop570:320:3:2")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':720, 'quality_height':320,
                                  'yuv_type':'yuv422p',})
        self.assertTrue(asset.crop_cmd is None)
        self.assertEquals(str(asset), "test_0_0__720x480_yuv422p_vs__720x480_yuv422p_q_720x320")

    def test_pad_cmd(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':720, 'quality_height':320,
                                  'yuv_type':'yuv422p',
                                  'pad_cmd':'iw+6:ih+4:3:2'})
        self.assertEquals(asset.pad_cmd, 'iw+6:ih+4:3:2')
        self.assertEquals(str(asset), "test_0_0__720x480_yuv422p_vs__720x480_yuv422p_q_720x320_padiw+6:ih+4:3:2")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width':720, 'height':480,
                                  'quality_width':720, 'quality_height':320,
                                  'yuv_type':'yuv422p',})
        self.assertTrue(asset.pad_cmd is None)
        self.assertEquals(str(asset), "test_0_0__720x480_yuv422p_vs__720x480_yuv422p_q_720x320")

if __name__ == '__main__':
    unittest.main()
