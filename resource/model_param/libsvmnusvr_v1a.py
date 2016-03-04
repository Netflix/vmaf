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

    # ==== libsvmnusvr parameters ==== #

    # 'gamma':0.0, # default
    'gamma':0.39, # selected

    'C':2.5, # default

    'nu':0.5, # default

}
