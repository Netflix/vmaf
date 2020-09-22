__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

from vmaf.config import VmafConfig
from vmaf.core.asset import Asset, NorefAsset


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


def set_default_576_324_10bit_videos_for_testing():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv422p10le.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv422p10le.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324,
                              'yuv_type': 'yuv422p10le'})

    asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           ref_path=ref_path,
                           dis_path=ref_path,
                           asset_dict={'width': 576, 'height': 324,
                                       'yuv_type': 'yuv422p10le'})

    return ref_path, dis_path, asset, asset_original


def set_default_576_324_10bit_videos_for_testing_b():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv420p10le.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv420p10le.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324,
                              'yuv_type': 'yuv420p10le'})

    asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           ref_path=ref_path,
                           dis_path=ref_path,
                           asset_dict={'width': 576, 'height': 324,
                                       'yuv_type': 'yuv420p10le'})

    return ref_path, dis_path, asset, asset_original


def set_default_576_324_12bit_videos_for_testing():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv420p12le.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv420p12le.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324,
                              'yuv_type': 'yuv420p12le'})

    asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           ref_path=ref_path,
                           dis_path=ref_path,
                           asset_dict={'width': 576, 'height': 324,
                                       'yuv_type': 'yuv420p12le'})

    return ref_path, dis_path, asset, asset_original


def set_default_576_324_16bit_videos_for_testing():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv420p16le.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv420p16le.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324,
                              'yuv_type': 'yuv420p16le'})

    asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           ref_path=ref_path,
                           dis_path=ref_path,
                           asset_dict={'width': 576, 'height': 324,
                                       'yuv_type': 'yuv420p16le'})

    return ref_path, dis_path, asset, asset_original


def set_default_576_324_noref_videos_for_testing():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
    asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324})

    asset_original = NorefAsset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           dis_path=ref_path,
                           asset_dict={'width': 576, 'height': 324})

    return ref_path, dis_path, asset, asset_original
