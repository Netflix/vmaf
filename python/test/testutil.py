__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset


def set_default_flat_1920_1080_videos_for_testing():
    ref_path = VmafConfig.test_resource_path("yuv", "flat_1920_1080_0.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "flat_1920_1080_10.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': 1920, 'height': 1080})

    asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           ref_path=ref_path,
                           dis_path=ref_path,
                           asset_dict={'width': 1920, 'height': 1080})

    return ref_path, dis_path, asset, asset_original


def set_default_576_324_videos_for_testing():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324})

    asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           ref_path=ref_path,
                           dis_path=ref_path,
                           asset_dict={'width': 576, 'height': 324})

    return ref_path, dis_path, asset, asset_original
