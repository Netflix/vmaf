dataset_name = 'example_image'

yuv_fmt = 'notyuv'
quality_width = 1920
quality_height = 1080
workfile_yuv_type = 'yuv444p'

from vmaf.config import VmafConfig

ref_videos = [
    {'content_id': 0, 'path': VmafConfig.resource_path('icpf', 'frame00000001.icpf')},

    {'content_id': 1, 'path': VmafConfig.resource_path('icpf', 'frame00000002.icpf')},
]

dis_videos = [
    {'content_id': 0, 'asset_id': 0, 'dmos': 100, 'path': VmafConfig.resource_path('icpf', 'frame00000000.icpf')}, # ref
    {'content_id': 0, 'asset_id': 1, 'dmos': 50,  'path': VmafConfig.resource_path('icpf', 'frame00000001.icpf')},

    {'content_id': 1, 'asset_id': 2, 'dmos': 100,  'path': VmafConfig.resource_path('icpf', 'frame00000002.icpf')}, # ref
    {'content_id': 1, 'asset_id': 3, 'dmos': 80,  'path': VmafConfig.resource_path('icpf', 'frame00000003.icpf')},
]


