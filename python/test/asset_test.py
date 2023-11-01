__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import unittest
import re

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset, NorefAsset


class AssetTest(unittest.TestCase):

    def test_workdir(self):
        import re
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={}, workdir_root="my_workdir_root")
        workdir = asset.workdir
        self.assertTrue(re.match(r"^my_workdir_root/[a-zA-Z0-9-]+$", workdir))

    def test_workdir_root(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={}, workdir_root="my_workdir_root")
        self.assertEqual(asset.workdir_root, "my_workdir_root")

    def test_ref_width_height(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_width': 1920, 'ref_height': 1080,})
        self.assertEqual(asset.ref_width_height, (1920, 1080))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_width': 1920, 'ref_height': 1080,
                                  'width': 720, 'height': 480})
        self.assertEqual(asset.ref_width_height, (1920, 1080))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480})
        self.assertEqual(asset.ref_width_height, (720, 480))

    def test_dis_width_height(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_width': 1920, 'dis_height': 1080, })
        self.assertEqual(asset.dis_width_height, (1920, 1080))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_width': 1920, 'dis_height': 1080,
                                  'width': 720, 'height': 480})
        self.assertEqual(asset.dis_width_height, (1920, 1080))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480})
        self.assertEqual(asset.dis_width_height, (720, 480))

    def test_dis_encode_width_height(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_width': 1920, 'dis_height': 1080,
                                  'dis_enc_width': 960, 'dis_enc_height': 540, })
        self.assertEqual(asset.dis_width_height, (1920, 1080))
        self.assertEqual(asset.dis_encode_width_height, (960, 540))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_width': 1920, 'dis_height': 1080,
                                  'width': 720, 'height': 480})
        self.assertEqual(asset.dis_encode_width_height, (1920, 1080))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480})
        self.assertEqual(asset.dis_encode_width_height, (720, 480))

    def test_dis_encode_bitdepth(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_enc_bitdepth': 8})
        self.assertEqual(asset.dis_encode_bitdepth, 8)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_enc_bitdepth': 24})
        with self.assertRaises(AssertionError):
            print(asset.dis_encode_bitdepth)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv420p'})
        self.assertEqual(asset.dis_encode_bitdepth, 8)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv422p'})
        self.assertEqual(asset.dis_encode_bitdepth, 8)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv444p'})
        self.assertEqual(asset.dis_encode_bitdepth, 8)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv420p10le'})
        self.assertEqual(asset.dis_encode_bitdepth, 10)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv422p10le'})
        self.assertEqual(asset.dis_encode_bitdepth, 10)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv444p10le'})
        self.assertEqual(asset.dis_encode_bitdepth, 10)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv420p12le'})
        self.assertEqual(asset.dis_encode_bitdepth, 12)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv422p12le'})
        self.assertEqual(asset.dis_encode_bitdepth, 12)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv444p12le'})
        self.assertEqual(asset.dis_encode_bitdepth, 12)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv420p16le'})
        self.assertEqual(asset.dis_encode_bitdepth, 16)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv422p16le'})
        self.assertEqual(asset.dis_encode_bitdepth, 16)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv444p16le'})
        self.assertEqual(asset.dis_encode_bitdepth, 16)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'dis_yuv_type': 'yuv420p10le', 'dis_enc_bitdepth': 8})
        self.assertEqual(asset.dis_encode_bitdepth, 8)

    def test_quality_width_height(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={"ref_width": 1920, "ref_height": 1080,
                                  "dis_width": 720, "dis_height": 480},)
        with self.assertRaises(AssertionError):
            print(asset.quality_width_height)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={"ref_width": 1920, "ref_height": 1080,
                                  "dis_width": 720, "dis_height": 480,
                                  "quality_width": 1280, "quality_height": 720},)
        self.assertEqual(asset.quality_width_height, (1280, 720))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={"ref_width": 720, "ref_height": 480,
                                  "dis_width": 720, "dis_height": 480, },)
        self.assertEqual(asset.quality_width_height, (720, 480))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={"width": 720, "height": 480, },)
        self.assertEqual(asset.quality_width_height, (720, 480))

    def test_start_end_frame(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_start_frame': 2, 'ref_end_frame': 2,
                                  'dis_start_frame': 3, 'dis_end_frame': 3},)
        self.assertEqual(asset.ref_start_end_frame, (2, 2))
        self.assertEqual(asset.dis_start_end_frame, (3, 3))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'start_frame': 2, 'end_frame': 2})
        self.assertEqual(asset.ref_start_end_frame, (2, 2))
        self.assertEqual(asset.dis_start_end_frame, (2, 2))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps':24, 'duration_sec': 2})
        self.assertEqual(asset.ref_start_end_frame, (0, 47))
        self.assertEqual(asset.dis_start_end_frame, (0, 47))

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps': 24, 'start_sec': 2, 'end_sec': 3})
        self.assertEqual(asset.ref_start_end_frame, (48, 71))
        self.assertEqual(asset.dis_start_end_frame, (48, 71))

    def test_duration_sec(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_start_frame': 2, 'ref_end_frame': 2,
                                  'dis_start_frame': 3, 'dis_end_frame': 3},)
        self.assertEqual(asset.ref_duration_sec, None)
        self.assertEqual(asset.dis_duration_sec, None)
        self.assertEqual(asset.ref_start_sec, None)
        self.assertEqual(asset.dis_start_sec, None)

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'ref_start_frame':  0, 'ref_end_frame': 23,
                                  'dis_start_frame': 3, 'dis_end_frame': 26,
                                  'fps':24},)
        self.assertEqual(asset.ref_duration_sec, 1.0)
        self.assertEqual(asset.dis_duration_sec, 1.0)
        self.assertAlmostEqual(asset.ref_start_sec, 0.0, places=4)
        self.assertAlmostEqual(asset.dis_start_sec, 0.125, places=4)

    def test_bitrate(self):
        ref_path = VmafConfig.test_resource_path('yuv/', 'src01_hrc00_576x324.yuv')
        dis_path = VmafConfig.test_resource_path('yuv', 'src01_hrc01_576x324.yuv')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path=ref_path, dis_path=dis_path,
                      asset_dict={'ref_start_frame': 0, 'ref_end_frame': 47,
                                  'dis_start_frame': 0, 'dis_end_frame': 47,
                                  'fps': 23.976},)
        self.assertAlmostEqual(asset.ref_bitrate_kbps_for_entire_file,
                          53693.964287999996, places=4)
        self.assertAlmostEqual(asset.dis_bitrate_kbps_for_entire_file,
                          53693.964287999996, places=4)

    def test_to_normalized_dict(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'start_frame': 2, 'end_frame': 2})
        self.assertEqual(
            asset.to_normalized_dict(),
            {
                'asset_dict': {'end_frame': 2, 'height': 480, 'start_frame': 2, 'width': 720},
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
                      asset_dict={'width': 720, 'height': 480,
                                  'start_frame': 2, 'end_frame': 2})
        self.assertEqual(
            asset.to_normalized_dict(),
            {
                'asset_dict': {'end_frame': 2, 'height': 480, 'start_frame': 2, 'width': 720},
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
                      asset_dict={'width': 720, 'height': 480,
                                  'start_frame': 2, 'end_frame': 2})
        self.assertEqual(
            str(asset),
            "test_0_0_refvideo_720x480_2to2_vs_disvideo_720x480_2to2_q_720x480"
        )
        expected_repr = '{"asset_dict": {"end_frame": 2, "height": 480, "start_frame": 2, "width": 720}, "asset_id": 0, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}' # noqa
        self.assertEqual(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEqual(asset, recon_asset)
        self.assertTrue(asset == recon_asset)
        self.assertFalse(asset != recon_asset)

        self.assertEqual(asset.to_normalized_repr(), expected_repr)

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480, })
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480"
        )
        expected_repr = '{"asset_dict": {"height": 480, "width": 720}, "asset_id": 1, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}'
        self.assertEqual(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEqual(asset, recon_asset)

        asset = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 1920, 'quality_height': 1080})
        self.assertEqual(
            str(asset),
            "test_0_2_refvideo_720x480_vs_disvideo_720x480_q_1920x1080"
        )
        expected_repr = '{"asset_dict": {"height": 480, "quality_height": 1080, "quality_width": 1920, "width": 720}, "asset_id": 2, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}' # noqa
        self.assertEqual(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEqual(asset, recon_asset)

        asset = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 1920, 'quality_height': 1080,
                                  'yuv_type': 'yuv422p'})
        self.assertEqual(
            str(asset),
            "test_0_2_refvideo_720x480_yuv422p_vs_disvideo_720x480_yuv422p_q_1920x1080"
        )
        expected_repr = '{"asset_dict": {"height": 480, "quality_height": 1080, "quality_width": 1920, "width": 720, "yuv_type": "yuv422p"}, "asset_id": 2, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}' # noqa
        self.assertEqual(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEqual(asset, recon_asset)

        asset = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 1920, 'quality_height': 1080,
                                  'resampling_type': 'lanczos'})
        self.assertEqual(
            str(asset),
            "test_0_2_refvideo_720x480_lanczos_vs_disvideo_720x480_lanczos_q_1920x1080"
        )
        expected_repr = '{"asset_dict": {"height": 480, "quality_height": 1080, "quality_width": 1920, "resampling_type": "lanczos", "width": 720}, "asset_id": 2, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}' # noqa
        self.assertEqual(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEqual(asset, recon_asset)

        asset = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'dis_enc_width': 1920, 'dis_enc_height': 1080})
        self.assertEqual(
            str(asset),
            "test_0_2_refvideo_720x480_vs_disvideo_720x480_e_1920x1080_q_720x480"
        )
        expected_repr = '{"asset_dict": {"dis_enc_height": 1080, "dis_enc_width": 1920, "height": 480, "width": 720}, "asset_id": 2, "content_id": 0, "dataset": "test", "dis_path": "disvideo.yuv", "ref_path": "refvideo.yuv", "workdir": ""}'  # noqa
        self.assertEqual(repr(asset), expected_repr)
        recon_asset = Asset.from_repr(expected_repr)
        self.assertEqual(asset, recon_asset)

    def test_str(self):

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 480})
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'resampling_type': 'lanczos'})
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 480,
                                  'resampling_type': 'lanczos'})
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'ref_width': 720, 'ref_height': 480,
                                  'dis_width': 1920, 'dis_height': 1080,
                                  'quality_width': 720, 'quality_height': 480,
                                  'resampling_type': 'lanczos'})
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_1920x1080_lanczos_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'ref_width': 1920, 'ref_height': 1080,
                                  'dis_width': 720, 'dis_height': 480,
                                  'quality_width': 720, 'quality_height': 480,
                                  'resampling_type': 'lanczos'})
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_1920x1080_lanczos_vs_disvideo_720x480_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'dis_enc_width': 480, 'dis_enc_height': 320})
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_e_480x320_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'dis_yuv_type': 'yuv420p10le', 'dis_enc_bitdepth': 10})
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_yuv420p10le_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'dis_yuv_type': 'yuv420p10le', 'dis_enc_bitdepth': 8})
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_ebd_8_yuv420p10le_q_720x480"
        )

        asset = Asset(dataset="test", content_id=0, asset_id=1,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'workfile_yuv_type': 'yuv420p10le'})
        self.assertEqual(
            str(asset),
            "test_0_1_refvideo_720x480_vs_disvideo_720x480_q_720x480_wf_yuv420p10le"
        )

    def test_hash_equal(self):
        asset1 = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 1920, 'quality_height': 1080})
        asset2 = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width':1920, 'quality_height': 1080})
        asset3 = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="my/dir/refvideo.yuv", dis_path="my/dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 1920, 'quality_height': 1080})
        asset4 = Asset(dataset="test", content_id=0, asset_id=2,
                      ref_path="my/dir/refvideo.yuv", dis_path="my/dir/disvideo.avi",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 1920, 'quality_height': 1080})

        self.assertTrue(asset1 == asset2)
        self.assertTrue(asset2 == asset3)
        self.assertFalse(asset3 == asset4)
        self.assertTrue(hash(asset2) == hash(asset3))
        self.assertFalse(hash(asset1) == hash(asset4))

    def test_workfile_path(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'start_frame': 2, 'end_frame': 2,
                                  'quality_width': 1920, 'quality_height': 1080},
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
                      asset_dict={'fps': 24, 'start_sec': 2, 'end_sec': 3})
        self.assertEqual(asset.ref_yuv_type, 'yuv420p')
        self.assertEqual(asset.dis_yuv_type, 'yuv420p')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="", asset_dict={
            'fps': 24, 'start_sec': 2, 'end_sec': 3, 'yuv_type':'yuv444p'})
        self.assertEqual(asset.ref_yuv_type, 'yuv444p')
        self.assertEqual(asset.dis_yuv_type, 'yuv444p')

        with self.assertRaises(AssertionError):
            asset = Asset(dataset="test", content_id=0, asset_id=0,
                          ref_path="", dis_path="",
                          asset_dict={'fps': 24, 'start_sec': 2,
                                      'end_sec': 3, 'yuv_type': 'yuv444a'})

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="", asset_dict={
                'fps': 24, 'start_sec': 2, 'end_sec': 3,
                'ref_yuv_type': 'notyuv', 'dis_yuv_type': 'yuv444p',
                'yuv_type': 'yuv420p' # redundant
            })
        self.assertEqual(asset.ref_yuv_type, 'notyuv')
        self.assertEqual(asset.dis_yuv_type, 'yuv444p')

    def test_resampling_type(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps': 24, 'start_sec': 2, 'end_sec': 3})
        self.assertEqual(asset.resampling_type, 'bicubic')
        self.assertEqual(asset.ref_resampling_type, 'bicubic')
        self.assertEqual(asset.dis_resampling_type, 'bicubic')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps': 24, 'start_sec': 2, 'end_sec': 3,
                                  'resampling_type': 'lanczos'})
        self.assertEqual(asset.resampling_type, 'lanczos')
        self.assertEqual(asset.ref_resampling_type, 'lanczos')
        self.assertEqual(asset.dis_resampling_type, 'lanczos')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps': 24, 'start_sec': 2, 'end_sec': 3,
                                  'resampling_type': 'bicubic'})
        self.assertEqual(asset.resampling_type, 'bicubic')
        self.assertEqual(asset.ref_resampling_type, 'bicubic')
        self.assertEqual(asset.dis_resampling_type, 'bicubic')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps': 24, 'start_sec': 2, 'end_sec': 3,
                                  'ref_resampling_type': 'bilinear',
                                  'dis_resampling_type': 'lanczos'})
        self.assertEqual(asset.ref_resampling_type, 'bilinear')
        self.assertEqual(asset.dis_resampling_type, 'lanczos')
        with self.assertRaises(AssertionError):
            _ = asset.resampling_type

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'fps': 24, 'start_sec': 2, 'end_sec': 3,
                                  'ref_resampling_type': 'bilinear',
                                  'dis_resampling_type': 'lanczos',
                                  'resampling_type': 'bilinear'})
        self.assertEqual(asset.ref_resampling_type, 'bilinear')
        self.assertEqual(asset.dis_resampling_type, 'lanczos')
        with self.assertRaises(AssertionError):
            _ = asset.resampling_type

    def test_use_path_as_workpath(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'start_frame': 2, 'end_frame': 2,
                                  'quality_width': 1920, 'quality_height': 1080},
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
        self.assertEqual(asset.ref_workfile_path, 'dir/refvideo.yuv')
        self.assertEqual(asset.dis_workfile_path, 'dir/disvideo.yuv')

    def test_crop_cmd(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'crop_cmd': '570:320:3:2'})
        self.assertEqual(asset.crop_cmd, '570:320:3:2')
        self.assertEqual(asset.ref_crop_cmd, '570:320:3:2')
        self.assertEqual(asset.dis_crop_cmd, '570:320:3:2')
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_crop570_320_3_2_vs_720x480_yuv422p_crop570_320_3_2_q_720x320")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p', })
        self.assertTrue(asset.crop_cmd is None)
        self.assertTrue(asset.ref_crop_cmd is None)
        self.assertTrue(asset.dis_crop_cmd is None)
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_vs_720x480_yuv422p_q_720x320")

    def test_ref_dis_crop_cmd(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'ref_crop_cmd': '570:320:3:2', 'dis_crop_cmd': '571:320:3:2'})
        self.assertTrue(asset.crop_cmd is None)
        self.assertEqual(asset.ref_crop_cmd, '570:320:3:2')
        self.assertEqual(asset.dis_crop_cmd, '571:320:3:2')
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_crop570_320_3_2_vs_720x480_yuv422p_crop571_320_3_2_q_720x320")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p', 'ref_crop_cmd': '570:320:3:2', 'crop_cmd': '572:320:3:2', })
        self.assertEqual(asset.crop_cmd, '572:320:3:2')
        self.assertEqual(asset.ref_crop_cmd, '570:320:3:2')
        self.assertEqual(asset.dis_crop_cmd, '572:320:3:2')
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_crop570_320_3_2_vs_720x480_yuv422p_crop572_320_3_2_q_720x320")

    def test_pad_cmd(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'pad_cmd': 'iw+6:ih+4:3:2'})
        self.assertEqual(asset.pad_cmd, 'iw+6:ih+4:3:2')
        self.assertEqual(asset.ref_pad_cmd, 'iw+6:ih+4:3:2')
        self.assertEqual(asset.dis_pad_cmd, 'iw+6:ih+4:3:2')
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_padiw_6_ih_4_3_2_vs_720x480_yuv422p_padiw_6_ih_4_3_2_q_720x320")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p', })
        self.assertTrue(asset.pad_cmd is None)
        self.assertTrue(asset.ref_pad_cmd is None)
        self.assertTrue(asset.pad_cmd is None)
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_vs_720x480_yuv422p_q_720x320")

    def test_ref_dis_pad_cmd(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'ref_pad_cmd': 'iw+6:ih+4:3:2'})
        self.assertIsNone(asset.pad_cmd)
        self.assertEqual(asset.ref_pad_cmd, 'iw+6:ih+4:3:2')
        self.assertIsNone(asset.dis_pad_cmd)
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_padiw_6_ih_4_3_2_vs_720x480_yuv422p_q_720x320")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'ref_pad_cmd': 'iw+6:ih+4:3:2', 'dis_pad_cmd': 'iw+6:ih+4:3:3'})
        self.assertIsNone(asset.pad_cmd)
        self.assertEqual(asset.ref_pad_cmd, 'iw+6:ih+4:3:2')
        self.assertEqual(asset.dis_pad_cmd, 'iw+6:ih+4:3:3')
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_padiw_6_ih_4_3_2_vs_720x480_yuv422p_padiw_6_ih_4_3_3_q_720x320")

    def test_notyuv(self):
        with self.assertRaises(AssertionError):
            asset = Asset(dataset="test", content_id=0, asset_id=0,
                          ref_path="", dis_path="",
                          asset_dict={
                              'yuv_type': 'notyuv',
                              'width': 720, 'height': 480,
                          })

        with self.assertRaises(AssertionError):
            asset = Asset(dataset="test", content_id=0, asset_id=0,
                          ref_path="", dis_path="",
                          asset_dict={
                              'yuv_type': 'notyuv',
                              'ref_width': 720, 'ref_height': 480,
                          })

        with self.assertRaises(AssertionError):
            asset = Asset(dataset="test", content_id=0, asset_id=0,
                          ref_path="", dis_path="",
                          asset_dict={
                              'yuv_type': 'notyuv',
                              'ref_width': 720, 'ref_height': 480,
                              'dis_width': 720, 'dis_height': 480,
                          })

        with self.assertRaises(AssertionError):
            asset = Asset(dataset="test", content_id=0, asset_id=0,
                          ref_path="", dis_path="",
                          asset_dict={
                              'yuv_type': 'notyuv',
                              'workfile_yuv_type': 'yuv4444p'
                          })

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={
                          'yuv_type': 'notyuv',
                          'workfile_yuv_type': 'yuv422p'
                      })
        self.assertEqual(asset.workfile_yuv_type, 'yuv422p')

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="refvideo", dis_path="disvideo",
                      asset_dict={
                          'yuv_type': 'notyuv',
                      })
        self.assertTrue(asset.quality_width_height is None)
        self.assertEqual(str(asset), "test_0_0_refvideo_notyuv_vs_disvideo_notyuv")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="refvideo", dis_path="disvideo",
                      asset_dict={
                          'yuv_type': 'notyuv',
                          'quality_width': 720, 'quality_height': 480,
                      })
        self.assertEqual(asset.quality_width_height, (720, 480))
        self.assertEqual(str(asset), "test_0_0_refvideo_notyuv_vs_disvideo_notyuv_q_720x480")

    def test_notyuv_noref(self):
        with self.assertRaises(AssertionError):
            asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                          dis_path="",
                          asset_dict={
                              'yuv_type': 'notyuv',
                              'workfile_yuv_type': 'yuv4444p'
                              })

        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      dis_path="",
                      asset_dict={
                          'yuv_type': 'notyuv',
                          'workfile_yuv_type': 'yuv422p'
                      })
        self.assertEqual(asset.workfile_yuv_type, 'yuv422p')

    def test_copy(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'crop_cmd': '570:320:3:2'})
        new_asset = asset.copy()
        self.assertEqual(asset, new_asset)
        self.assertTrue(asset == new_asset)
        self.assertNotEqual(id(asset), id(new_asset))

        new_asset.asset_dict['yuv_type'] = 'yuv444p'
        self.assertNotEqual(asset, new_asset)

        new_asset2 = asset.copy(content_id=2)
        self.assertFalse(asset == new_asset2)
        self.assertEqual(new_asset2.content_id, 2)
        self.assertEqual(new_asset2.asset_id, 0)

    def test_copy_use_path_as_workpath(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'crop_cmd': '570:320:3:2',
                                  'use_path_as_workpath': True
                                  })
        new_asset = asset.copy()
        self.assertNotEqual(asset, new_asset)  # use_path_as_workpath gets reset
        self.assertTrue(asset.use_path_as_workpath)
        self.assertFalse(new_asset.use_path_as_workpath)

    def test_copy_override_asset_dict(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'crop_cmd': '570:320:3:2'})
        new_asset = asset.copy(asset_dict={'height': 481})
        self.assertEqual(new_asset.ref_width_height, (720, 481))
        self.assertEqual(new_asset.dis_width_height, (720, 481))
        self.assertEqual(new_asset.ref_crop_cmd, '570:320:3:2')
        self.assertEqual(new_asset.dis_crop_cmd, '570:320:3:2')

    def test_copy_additional_asset_dict(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'crop_cmd': '570:320:3:2'})
        new_asset = asset.copy(asset_dict={'start_frame': 1, 'end_frame': 10})
        self.assertEqual(new_asset.ref_start_end_frame, (1, 10))
        self.assertEqual(new_asset.dis_start_end_frame, (1, 10))
        self.assertEqual(new_asset.ref_width_height, (720, 480))
        self.assertEqual(new_asset.dis_width_height, (720, 480))
        self.assertEqual(new_asset.dis_crop_cmd, '570:320:3:2')
        self.assertEqual(new_asset.ref_crop_cmd, '570:320:3:2')

    def test_copy_noref(self):
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'crop_cmd': '570:320:3:2'})
        new_asset = asset.copy()
        self.assertEqual(asset, new_asset)
        self.assertTrue(asset == new_asset)
        self.assertNotEqual(id(asset), id(new_asset))

        new_asset.asset_dict['yuv_type'] = 'yuv444p'
        self.assertNotEqual(asset, new_asset)

        new_asset2 = asset.copy(content_id=2)
        self.assertFalse(asset == new_asset2)
        self.assertEqual(new_asset2.content_id, 2)
        self.assertEqual(new_asset2.asset_id, 0)

    def test_NorefAsset_copy_as_Asset(self):
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      dis_path="abc",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'crop_cmd': '570:320:3:2'})
        new_asset = asset.copy(ref_path='xyz')
        self.assertEqual(new_asset.ref_path, 'abc')
        new_asset2 = asset.copy_as_asset(ref_path='xyz')
        self.assertEqual(new_asset2.ref_path, 'xyz')

    def test_NorefAsset_copy_override_asset_dict(self):
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                           dis_path="",
                           asset_dict={'width': 720, 'height': 480,
                                       'quality_width': 720, 'quality_height': 320,
                                       'yuv_type': 'yuv422p',
                                       'crop_cmd': '570:320:3:2'})
        new_asset = asset.copy(asset_dict={'height': 481})
        self.assertEqual(new_asset.dis_width_height, (720, 481))
        self.assertEqual(new_asset.dis_crop_cmd, '570:320:3:2')

    def test_clear_up_yuv_type(self):
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      dis_path="abc",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'crop_cmd': '570:320:3:2'})
        self.assertEqual(asset.dis_yuv_type, 'yuv422p')
        asset.clear_up_yuv_type()
        self.assertEqual(asset.dis_yuv_type, 'yuv420p')

    def test_clear_up_width_height(self):
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      dis_path="abc",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'crop_cmd': '570:320:3:2'})
        self.assertEqual(asset.dis_width_height, (720, 480))
        asset.clear_up_width_height()
        self.assertEqual(asset.dis_width_height, None)

    def test_clear_up_start_end_frame(self):
        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      dis_path="abc",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'start_frame': 23, 'end_frame': 25})
        self.assertEqual(asset.dis_start_end_frame, (23, 25))
        asset.clear_up_start_end_frame()
        self.assertEqual(asset.dis_start_end_frame, None)

        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      dis_path="abc",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'fps': 5,
                                  'duration_sec': 10})
        self.assertEqual(asset.dis_start_end_frame, (0, 49))
        asset.clear_up_start_end_frame()
        self.assertEqual(asset.dis_start_end_frame, None)

    def test_encoder_output_asset(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="abc.yuv",
                      dis_path="abc10to14",
                      asset_dict={'dis_yuv_type': 'notyuv',
                                  'fps': 23.976,
                                  'ref_end_frame': 14,
                                  'ref_height': 324,
                                  'ref_start_frame': 10,
                                  'ref_width': 576,
                                  'ref_yuv_type': 'yuv420p'})
        self.assertEqual(str(asset), "test_0_0_abc_576x324_10to14_vs_abc10to14_notyuv_q_576x324")
        self.assertEqual(asset.ref_yuv_type, "yuv420p")
        self.assertEqual(asset.dis_yuv_type, "notyuv")
        self.assertEqual(asset.ref_width_height, (576, 324))
        self.assertEqual(asset.dis_width_height, None)
        self.assertEqual(asset.ref_start_end_frame, (10, 14))
        self.assertEqual(asset.dis_start_end_frame, None)

    def test_groundtruth(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="dir/refvideo.yuv", dis_path="dir/disvideo.yuv",
                      asset_dict={'width': 720, 'height': 480,
                                  'start_frame': 2, 'end_frame': 2,
                                  'quality_width': 1920, 'quality_height': 1080,
                                  'groundtruth': 91.0, 'groundtruth_std': 4.5, 'raw_groundtruth': [90.0, 92.0]},
                      workdir_root="workdir")
        self.assertEqual(asset.groundtruth, 91.0)
        self.assertEqual(asset.groundtruth_std, 4.5)
        self.assertEqual(asset.raw_groundtruth, [90.0, 92.0])

    def test_gblur_cmd(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'gblur_cmd': 'sigma=1:steps=2'})
        self.assertEqual(asset.get_filter_cmd('gblur'), 'sigma=1:steps=2')
        self.assertEqual(asset.get_filter_cmd('gblur', 'ref'), 'sigma=1:steps=2')
        self.assertEqual(asset.get_filter_cmd('gblur', 'dis'), 'sigma=1:steps=2')
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_gblursigma_1_steps_2_vs_720x480_yuv422p_gblursigma_1_steps_2_q_720x320")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p', })
        self.assertTrue(asset.get_filter_cmd('gblur') is None)
        self.assertTrue(asset.get_filter_cmd('gblur', 'ref') is None)
        self.assertTrue(asset.get_filter_cmd('gblur', 'dis') is None)
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_vs_720x480_yuv422p_q_720x320")

    def test_ref_dis_gblur_cmd(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'ref_gblur_cmd': 'sigma=1:steps=2',
                                  'dis_gblur_cmd': 'sigma=0.3:steps=3',
                                  })
        self.assertEqual(asset.get_filter_cmd('gblur'), None)
        self.assertEqual(asset.get_filter_cmd('gblur', 'ref'), 'sigma=1:steps=2')
        self.assertEqual(asset.get_filter_cmd('gblur', 'dis'), 'sigma=0.3:steps=3')
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_gblursigma_1_steps_2_vs_720x480_yuv422p_gblursigma_0_3_steps_3_q_720x320")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path="", dis_path="",
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'ref_gblur_cmd': 'sigma=1:steps=2',
                                  'gblur_cmd': 'sigma=0.3:steps=3',
                                  })
        self.assertEqual(asset.get_filter_cmd('gblur'), 'sigma=0.3:steps=3')
        self.assertEqual(asset.get_filter_cmd('gblur', 'ref'), 'sigma=1:steps=2')
        self.assertEqual(asset.get_filter_cmd('gblur', 'dis'), 'sigma=0.3:steps=3')
        self.assertEqual(str(asset), "test_0_0_720x480_yuv422p_gblursigma_1_steps_2_vs_720x480_yuv422p_gblursigma_0_3_steps_3_q_720x320")

    def test_long_string(self):
        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path='a'*128, dis_path='b'*128,
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'ref_pad_cmd': 'iw+6:ih+4:3:2'})
        self.assertEqual(str(asset), "4a67755df3b8c599c9016a6d03171392ddfbf9c2")

        asset = Asset(dataset="test", content_id=0, asset_id=0,
                      ref_path='a'*60, dis_path='b'*60,
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'ref_pad_cmd': 'iw+6:ih+4:3:2'})
        self.assertEqual(str(asset), "test_0_0_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_720x480_yuv422p_padiw_6_ih_4_3_2_vs_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb_720x480_yuv422p_q_720x320")

        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      dis_path='a'*128,
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'ref_pad_cmd': 'iw+6:ih+4:3:2'})
        self.assertEqual(str(asset), "test_0_0_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_720x480_yuv422p_q_720x320")

        asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                      dis_path='b'*60,
                      asset_dict={'width': 720, 'height': 480,
                                  'quality_width': 720, 'quality_height': 320,
                                  'yuv_type': 'yuv422p',
                                  'ref_pad_cmd': 'iw+6:ih+4:3:2',  # no effect
                                  })
        self.assertEqual(str(asset), "test_0_0_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb_720x480_yuv422p_q_720x320")

    def test_crop_pad_special(self):
        import os
        asset = \
            Asset(
                dataset='vp9eve_vmaf_investigation', content_id=0, asset_id=0,
                ref_path=os.path.join('dir', 'inception', 'Inception_A__79_41_79_46__1920_1080__', 'frame%08d.icpf'),
                dis_path=os.path.join('dir', 'inception', '10014993632.ivf'),
                asset_dict={
                    'ref_yuv_type': 'notyuv', 'ref_crop_cmd': '1920:800:0:140', 'ref_pad_cmd': 'iw+0:ih+280:0:140', 'ref_start_frame': 0, 'ref_end_frame': 2,
                    'dis_yuv_type': 'notyuv', 'dis_start_frame': 114629, 'dis_end_frame': 114629 + 2,
                    'quality_width': 1920, 'quality_height': 1080, 'resampling_type': 'bicubic',
                    'workfile_yuv_type': 'yuv420p',
                })
        self.assertEqual(os.path.basename(asset.ref_workfile_path), "ref_vp9eve_vmaf_investigation_0_0_frame_08d_notyuv_0to2_crop1920_800_0_140_padiw_0_ih_280_0_140_vs_10014993632_notyuv_114629to114631_q_1920x1080")
        self.assertEqual(os.path.basename(asset.dis_workfile_path), "dis_vp9eve_vmaf_investigation_0_0_frame_08d_notyuv_0to2_crop1920_800_0_140_padiw_0_ih_280_0_140_vs_10014993632_notyuv_114629to114631_q_1920x1080")


if __name__ == '__main__':
    unittest.main(verbosity=2)
