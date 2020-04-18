dataset_name = 'example'

quality_width = 1920
quality_height = 1080

from vmaf.config import VmafConfig

ref_videos = [
    {'content_id': 0, 'path': VmafConfig.resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv'), 'yuv_fmt': 'yuv420p'},
]

dis_videos = [
    {'content_id': 0, 'asset_id': 0, 'dmos': 100, 'path': VmafConfig.resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv'), 'fps': 30, 'rebuf_indices': [0, -4, 15]}, # ref, make up fps and rebuf indices
]