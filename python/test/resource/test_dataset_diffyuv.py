dataset_name = 'example'

from vmaf.config import VmafConfig

ref_videos = [
    {'content_id': 0, 'path': VmafConfig.resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv'), 'width': 1920, 'height': 1080, 'yuv_fmt': 'yuv420p'},

    {'content_id': 1, 'path': VmafConfig.resource_path('yuv', 'flat_1280_720_0.yuv'), 'width': 1280, 'height': 720, 'yuv_fmt': 'yuv420p10le'},
]

dis_videos = [
    {'content_id': 0, 'asset_id': 0, 'groundtruth': 100, 'path': VmafConfig.resource_path('yuv', 'checkerboard_1920_1080_10_3_0_0.yuv')}, # ref
    {'content_id': 0, 'asset_id': 1, 'groundtruth': 50, 'path': VmafConfig.resource_path('yuv', 'checkerboard_1920_1080_10_3_1_0.yuv')},

    {'content_id': 1, 'asset_id': 2, 'groundtruth': 100, 'path': VmafConfig.resource_path('yuv', 'flat_1280_720_0.yuv')}, # ref
    {'content_id': 1, 'asset_id': 3, 'groundtruth': 80, 'path': VmafConfig.resource_path('yuv', 'flat_1280_720_10.yuv')},
]
