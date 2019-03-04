dataset_name = 'example'

yuv_fmt = 'yuv420p'
width = 576
height = 324
quality_width = 576
quality_height = 324
duration_sec = 5.0

from vmaf.config import VmafConfig

ref_videos = [
    {'content_id': 0, 'path': VmafConfig.test_resource_path('yuv', 'src01_hrc00_576x324.yuv')},
]

dis_videos = [
    {'content_id': 0, 'asset_id': 1, 'dmos': 50, 'path': VmafConfig.test_resource_path('yuv', 'src01_hrc01_576x324.yuv'), 'crop_cmd': '288:162:144:81'},
    {'content_id': 0, 'asset_id': 2, 'dmos': 49, 'path': VmafConfig.test_resource_path('yuv', 'src01_hrc01_576x324.yuv'), 'pad_cmd': 'iw+100:ih+100:50:50'},
    {'content_id': 0, 'asset_id': 3, 'dmos': 48, 'path': VmafConfig.test_resource_path('yuv', 'src01_hrc01_576x324.yuv'), 'crop_cmd': '288:162:144:81', 'pad_cmd': 'iw+288:ih+162:144:81',},
]
