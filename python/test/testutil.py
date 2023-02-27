__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import unittest

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

def set_default_576_324_videos_for_testing_workfile_yuv_10b():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324, 'workfile_yuv_type': 'yuv420p10le'})

    asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           ref_path=ref_path,
                           dis_path=ref_path,
                           asset_dict={'width': 576, 'height': 324, 'workfile_yuv_type': 'yuv420p10le'})

    return ref_path, dis_path, asset, asset_original

def set_default_576_324_videos_for_testing_scaled():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324,
                              'dis_enc_width': 480, 'dis_enc_height': 270})

    asset_original = Asset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           ref_path=ref_path,
                           dis_path=ref_path,
                           asset_dict={'width': 576, 'height': 324,
                                       'dis_enc_width': 480, 'dis_enc_height': 270})

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

def set_default_576_324_noref_videos_for_testing_workfile_yuv_10b():
    ref_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv")
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc01_576x324.yuv")
    asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324, 'workfile_yuv_type': 'yuv420p10le'})

    asset_original = NorefAsset(dataset="test", content_id=0, asset_id=1,
                           workdir_root=VmafConfig.workdir_path(),
                           dis_path=ref_path,
                           asset_dict={'width': 576, 'height': 324, 'workfile_yuv_type': 'yuv420p10le'})

    return ref_path, dis_path, asset, asset_original

def set_default_cambi_video_for_testing():
    dis_path = VmafConfig.test_resource_path("yuv", "blue_sky_360p_60f.yuv")
    asset = NorefAsset(dataset="test", content_id=0, asset_id=0,
                       workdir_root=VmafConfig.workdir_path(),
                       dis_path=dis_path,
                       asset_dict={'width': 640, 'height': 360, 'yuv_type': 'yuv420p'})

    return dis_path, asset


def set_default_cambi_video_for_testing_b():
    dis_path = VmafConfig.test_resource_path("yuv", "KristenAndSara_1280x720_8bit_processed.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=dis_path,
                  dis_path=dis_path,
                  asset_dict={'width': 1280, 'height': 720,
                              'dis_enc_width': 960, 'dis_enc_height': 540})

    return dis_path, dis_path, asset, asset

def set_default_cambi_video_for_testing_10b():
    dis_path = VmafConfig.test_resource_path("yuv", "src01_hrc00_576x324.yuv420p10le.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=dis_path,
                  dis_path=dis_path,
                  asset_dict={'width': 576, 'height': 324,
                              'yuv_type': 'yuv420p10le'})

    return dis_path, dis_path, asset, asset
