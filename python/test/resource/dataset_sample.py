dataset_name = 'example'

yuv_fmt = 'yuv420p'
width = 1920
height = 1080

from vmaf.config import VmafConfig

ref_videos = [
    {'content_id': 0, 'path': VmafConfig.test_resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv')},

    {'content_id': 1, 'path': VmafConfig.test_resource_path('yuv', 'flat_1920_1080_0.yuv')},
]

dis_videos = [
    {'content_id': 0, 'asset_id': 0, 'dmos': 100, 'path': VmafConfig.test_resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv')}, # ref
    {'content_id': 0, 'asset_id': 1, 'dmos': 50, 'path': VmafConfig.test_resource_path('yuv', 'checkerboard_1920_1080_10_3_1_0.yuv')},

    {'content_id': 1, 'asset_id': 2, 'dmos': 100, 'path': VmafConfig.test_resource_path('yuv', 'flat_1920_1080_0.yuv')}, # ref
    {'content_id': 1, 'asset_id': 3, 'dmos': 80, 'path': VmafConfig.test_resource_path('yuv', 'flat_1920_1080_10.yuv')},
]
