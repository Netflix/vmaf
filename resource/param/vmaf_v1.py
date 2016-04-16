feature_dict = {

    'VMAF_feature': ['vif', 'adm', 'motion', 'ansnr'],

}

model_type = "LIBSVMNUSVR"
model_param_dict = {

    # ==== preprocess: normalize each feature ==== #
    # 'norm_type': 'none', # default: do nothing
    'norm_type': 'clip_0to1', # rescale to within [0, 1]
    # 'norm_type': 'clip_minus1to1', # rescale to within [-1, 1]
    # 'norm_type': 'normalize', # rescale to mean zero and std one

    # ==== postprocess: clip final quality score ==== #
    # 'score_clip': None, # default: do nothing
    'score_clip': [0.0, 100.0], # clip to within [0, 100]

    # ==== libsvmnusvr parameters ==== #

    # 'gamma': 0.0, # default
    'gamma': 0.1, # vmafv1

    # 'C': 1.0, # default
    'C': 3.1, # vmafv1

    # 'nu': 0.5, # default
    'nu': 0.9, # vmafv1
}
