dataset_name = 'example'

import config

ref_videos = [
    {'content_id':0, 'path':config.ROOT+'/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv', 'width': 1920, 'height': 1080, 'yuv_fmt': 'yuv420p'},

    {'content_id':1, 'path':config.ROOT+'/resource/yuv/flat_1280_720_0.yuv', 'width': 1280, 'height': 720, 'yuv_fmt': 'yuv420p10le'},
]

dis_videos = [
    {'content_id':0, 'asset_id': 0, 'groundtruth':100, 'path':config.ROOT+'/resource/yuv/checkerboard_1920_1080_10_3_0_0.yuv'}, # ref
    {'content_id':0, 'asset_id': 1, 'groundtruth':50,  'path':config.ROOT+'/resource/yuv/checkerboard_1920_1080_10_3_1_0.yuv'},

    {'content_id':1, 'asset_id': 2, 'groundtruth':100,  'path':config.ROOT+'/resource/yuv/flat_1280_720_0.yuv'}, # ref
    {'content_id':1, 'asset_id': 3, 'groundtruth':80,  'path':config.ROOT+'/resource/yuv/flat_1280_720_10.yuv'},
]
