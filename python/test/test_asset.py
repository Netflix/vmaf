__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import unittest

from asset import Asset

class AssetTest(unittest.TestCase):

    def test_workdir(self):
        import re
        asset = Asset(ref_path="", dis_path="", asset_dict={},
                      workdir_root="my_workdir_root")
        workdir = asset.workdir
        mo = re.match(r"my_workdir_root/[a-zA-Z0-9]+", workdir)
        self.assertTrue(mo is not None)

    def test_ref_width_height(self):
        asset = Asset(ref_path="", dis_path="",
                      asset_dict={'ref_width':1920, 'ref_height':1080,})
        self.assertEquals(asset.ref_width_height, (1920, 1080))

        asset = Asset(ref_path="", dis_path="",
                      asset_dict={'ref_width':1920, 'ref_height':1080,
                                  'width':720, 'height':480})
        self.assertEquals(asset.ref_width_height, (1920, 1080))

        asset = Asset(ref_path="", dis_path="",
                      asset_dict= {'width':720, 'height':480})
        self.assertEquals(asset.ref_width_height, (720, 480))

    def test_dis_width_height(self):
        asset = Asset(ref_path="", dis_path="",
                      asset_dict={'dis_width':1920, 'dis_height':1080,})
        self.assertEquals(asset.dis_width_height, (1920, 1080))

    def test_start_end_frame(self):
        asset = Asset(ref_path="", dis_path="",
                      asset_dict={'start_frame':2, 'end_frame':2})
        self.assertEquals(asset.start_end_frame, (2, 2))

        asset = Asset(ref_path="", dis_path="",
                      asset_dict={'fps':24, 'duration_sec':2})
        self.assertEquals(asset.start_end_frame, (0, 47))

        asset = Asset(ref_path="", dis_path="",
                      asset_dict={'fps':24, 'start_sec':2, 'end_sec': 3})
        self.assertEquals(asset.start_end_frame, (48, 71))

if __name__ == '__main__':
    unittest.main()
