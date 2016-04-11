model_type = "LIBSVMNUSVR"
model_param_dict = {

    # ==== preprocess: normalize each feature ==== #
    'norm_type':'clip_0to1', # rescale to within [0, 1]

    # ==== postprocess: clip final quality score ==== #

    'score_clip':[0.0, 100.0], # clip to within [0, 100]

}
