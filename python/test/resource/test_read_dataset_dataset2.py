import os

dataset_name = 'DBName'
quality_width = 1280
quality_height = 1920
fps = 1
duration_sec = 10

root_dir = ""

ref_videos = [
    {'content_id': 0, 'content_name': 'XXX360', 'path': os.path.join(root_dir, 'XXX360Ref.mkv'),
     'yuv_fmt': 'notyuv'}
]

dis_videos = [

    {'asset_id': 3, 'content_id': 0, 'path': os.path.join(root_dir, 'XXX360Ref.mkv'), 'yuv_fmt': 'notyuv',
     'groundtruth': None, 'crop_cmd': '1280:1920:0:0'},

    {'asset_id': 4, 'content_id': 0, 'path': os.path.join(root_dir, 'XXX360DisScaling.mp4'), 'yuv_fmt': 'notyuv',
     'groundtruth': None, 'crop_cmd': '1280:1920:0:0', 'start_frame': 100, 'end_frame': 110},

    {'asset_id': 5, 'content_id': 0, 'path': os.path.join(root_dir, 'XXX360DisTiling.mp4'), 'yuv_fmt': 'notyuv',
     'groundtruth': None, 'crop_cmd': '1280:1920:0:0'},

]
