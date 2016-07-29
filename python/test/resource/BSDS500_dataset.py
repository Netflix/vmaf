import config

dataset_name = 'BSDS500'
yuv_fmt = 'yuv444p'

dataset_dir = config.ROOT + '/python/test/resource/BSDS500_yuv'

ref_videos = [{'content_id': 0,
  'content_name': '100007',
  'height': 321,
  'path': dataset_dir + '/100007.yuv',
  'width': 481},
 {'content_id': 1,
  'content_name': '100039',
  'height': 321,
  'path': dataset_dir + '/100039.yuv',
  'width': 481},
 {'content_id': 2,
  'content_name': '100075',
  'height': 321,
  'path': dataset_dir + '/100075.yuv',
  'width': 481},
 {'content_id': 3,
  'content_name': '100080',
  'height': 481,
  'path': dataset_dir + '/100080.yuv',
  'width': 321},
 {'content_id': 4,
  'content_name': '100098',
  'height': 321,
  'path': dataset_dir + '/100098.yuv',
  'width': 481}]

dis_videos = [{'asset_id': 0,
  'content_id': 0,
  'path': dataset_dir + '/100007.yuv', 'mos': 1},
 {'asset_id': 1,
  'content_id': 1,
  'path': dataset_dir + '/100039.yuv', 'mos': 2},
 {'asset_id': 2,
  'content_id': 2,
  'path': dataset_dir + '/100075.yuv', 'mos': 3},
 {'asset_id': 3,
  'content_id': 3,
  'path': dataset_dir + '/100080.yuv', 'mos': 4},
 {'asset_id': 4,
  'content_id': 4,
  'path': dataset_dir + '/100098.yuv', 'mos': 5}]
