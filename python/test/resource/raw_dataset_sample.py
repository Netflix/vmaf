dataset_name = 'example'

yuv_fmt = 'yuv420p'
width = 1920
height = 1080
ref_score = 100.0

from vmaf.config import VmafConfig

ref_videos = [
    {'content_id': 0, 'path': VmafConfig.test_resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv')},

    {'content_id': 1, 'path': VmafConfig.test_resource_path('yuv', 'flat_1920_1080_0.yuv')},
]

dis_videos = [
    {'content_id': 0, 'asset_id': 0, 'os': [100, 100, 100, 100, 100], 'path': VmafConfig.test_resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv')}, # ref
    {'content_id': 0, 'asset_id': 1, 'os': [40, 45, 50, 55, 60],  'path': VmafConfig.test_resource_path('yuv', 'checkerboard_1920_1080_10_3_1_0.yuv')},

    {'content_id': 1, 'asset_id': 2, 'os': [90, 90, 90, 90, 90],  'path': VmafConfig.test_resource_path('yuv', 'flat_1920_1080_0.yuv')}, # ref
    {'content_id': 1, 'asset_id': 3, 'os': [70, 75, 80, 85, 90],  'path': VmafConfig.test_resource_path('yuv', 'flat_1920_1080_10.yuv')},
]
