model_type = "LIBSVMNUSVR"
model_param_dict = {

    # ==== preprocess: normalize each feature ==== #
    'norm_type':'clip_0to1', # rescale to within [0, 1]

    # ==== postprocess: clip final quality score ==== #

    'score_clip':[0.0, 100.0], # clip to within [0, 100]

    'dis1st_thr':30.0, # wrap score towards score_clip[1] if luma <= dis1st_thr,
                       # active when score_clip is specified and
                       # Moment:dis1st (pixel mean) is one of the features
                       # specified in feature_param

}
