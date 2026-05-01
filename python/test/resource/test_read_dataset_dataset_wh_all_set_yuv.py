import os

dataset_name = 'DBName'
quality_width = 1920
quality_height = 1080

root_dir = ""

ref_videos = [
    {'content_id': 0, 'content_name': 'a', 'path': os.path.join(root_dir, 'a.mp4'), 'yuv_fmt': 'yuv420p10le',
     'width': 1919, 'height': 1081},
]

dis_videos = [

    {'asset_id': 0, 'content_id': 0, 'path': os.path.join(root_dir, 'a.mp4'), 'yuv_fmt': 'yuv420p10le',
     'width': 1921, 'height': 1079},

]
