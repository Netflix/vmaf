dataset_name = 'example'

quality_width = 1920
quality_height = 1080

from vmaf.config import VmafConfig

ref_videos = [
    {'content_id': 0, 'path': VmafConfig.resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv'), 'yuv_fmt': 'yuv420p', 'width': 1920, 'height': 1080},

    {'content_id': 1, 'path': VmafConfig.resource_path('yuv', 'flat_1920_1080_0.yuv'), 'yuv_fmt': 'yuv420p', 'width': 720, 'height': 480},
]

dis_videos = [
    {'content_id': 0, 'asset_id': 0, 'dmos': 100, 'path': VmafConfig.resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv'), 'yuv_fmt': 'yuv420p', 'width': 1920, 'height': 1080}, # ref
    {'content_id': 0, 'asset_id': 1, 'dmos': 50,  'path': VmafConfig.resource_path('yuv', 'checkerboard_1920_1080_10_3_1_0.264'), 'yuv_fmt': 'notyuv',},

    {'content_id': 1, 'asset_id': 2, 'dmos': 100,  'path': VmafConfig.resource_path('yuv', 'flat_1920_1080_0.yuv'), 'yuv_fmt': 'yuv420p', 'width': 720, 'height': 480}, # ref
    {'content_id': 1, 'asset_id': 3, 'dmos': 80,  'path': VmafConfig.resource_path('yuv', 'flat_1920_1080_10.264'), 'yuv_fmt': 'notyuv',},
]
