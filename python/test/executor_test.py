import unittest
from vmaf.core.asset import Asset
from vmaf.core.executor import Executor

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class ExecutorTest(unittest.TestCase):

    def test_get_workfile_yuv_type(self):

        asset = Asset(dataset="test", content_id=0, asset_id=0, ref_path="", dis_path="",
                      asset_dict={}, workdir_root="my_workdir_root")
        self.assertEqual(Executor._get_workfile_yuv_type(asset), 'yuv420p')

        asset = Asset(dataset="test", content_id=0, asset_id=0, ref_path="", dis_path="",
                      asset_dict={'ref_yuv_type': 'notyuv', 'dis_yuv_type': 'notyuv'}, workdir_root="my_workdir_root")
        self.assertEqual(Executor._get_workfile_yuv_type(asset), 'yuv420p')

        asset = Asset(dataset="test", content_id=0, asset_id=0, ref_path="", dis_path="",
                      asset_dict={'ref_yuv_type': 'yuv444p', 'dis_yuv_type': 'notyuv'}, workdir_root="my_workdir_root")
        self.assertEqual(Executor._get_workfile_yuv_type(asset), 'yuv444p')

        with self.assertRaises(AssertionError):
            asset = Asset(dataset="test", content_id=0, asset_id=0, ref_path="", dis_path="",
                          asset_dict={'ref_yuv_type': 'yuv444p', 'dis_yuv_type': 'yuv420p'}, workdir_root="my_workdir_root")
            self.assertEqual(Executor._get_workfile_yuv_type(asset), 'yuv444p')

        asset = Asset(dataset="test", content_id=0, asset_id=0, ref_path="", dis_path="",
                      asset_dict={'ref_yuv_type': 'notyuv', 'dis_yuv_type': 'yuv422p'}, workdir_root="my_workdir_root")
        self.assertEqual(Executor._get_workfile_yuv_type(asset), 'yuv422p')

        asset = Asset(dataset="test", content_id=0, asset_id=0, ref_path="", dis_path="",
                      asset_dict={'ref_yuv_type': 'yuv444p', 'dis_yuv_type': 'yuv444p'}, workdir_root="my_workdir_root")
        self.assertEqual(Executor._get_workfile_yuv_type(asset), 'yuv444p')

        asset = Asset(dataset="test", content_id=0, asset_id=0, ref_path="", dis_path="",
                      asset_dict={'ref_yuv_type': 'yuv444p', 'dis_yuv_type': 'yuv444p', 'workfile_yuv_type': 'yuv420p10le'}, workdir_root="my_workdir_root")
        self.assertEqual(Executor._get_workfile_yuv_type(asset), 'yuv420p10le')

        asset = Asset(dataset="test", content_id=0, asset_id=0, ref_path="", dis_path="",
                      asset_dict={'ref_yuv_type': 'yuv444p', 'dis_yuv_type': 'notyuv', 'workfile_yuv_type': 'yuv420p'}, workdir_root="my_workdir_root")
        self.assertEqual(Executor._get_workfile_yuv_type(asset), 'yuv420p')

        asset = Asset(dataset="test", content_id=0, asset_id=0, ref_path="", dis_path="",
                      asset_dict={'ref_yuv_type': 'yuv444p', 'dis_yuv_type': 'notyuv', 'workfile_yuv_type': 'yuv444p'}, workdir_root="my_workdir_root")
        self.assertEqual(Executor._get_workfile_yuv_type(asset), 'yuv444p')


if __name__ == "__main__":
    unittest.main(verbosity=2)
