model_type = "LIBSVMNUSVR"
model_param_dict = {

    # ==== preprocess: normalize each feature ==== #
    # 'norm_type':'none', # default: do nothing
    'norm_type':'clip_0to1', # rescale to within [0, 1]
    # 'norm_type':'clip_minus1to1', # rescale to within [-1, 1]
    # 'norm_type':'normalize', # rescale to mean zero and std one

    # ==== postprocess: clip final quality score ==== #

    # 'score_clip':None, # default: do nothing
    'score_clip':[0.0, 100.0], # clip to within [0, 100]

    'dis1st_thr':40.0, # wrap score towards score_clip[1] if luma <= dis1st_thr,
                       # active when score_clip is specified and
                       # Moment:dis1st (pixel mean) is one of the features
                       # specified in feature_param

    # ==== libsvmnusvr parameters ==== #

    # 'gamma':0.0, # default
    'gamma':0.6, # selected

    # 'C':1.0, # default
    'C':1.5, # default

    'nu':0.5, # default

    'cache_size':200 # default
}
