feature_dict = {

   'VMAF_feature': ['vif_scale0', 'vif_scale1', 'vif_scale2', 'vif_scale3', 'adm2', 'motion',],

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
    'gamma': 0.05, # vmaf_v3, vmaf_v4

    # 'C': 1.0, # default
    'C': 4.0, # vmaf_v4

    # 'nu': 0.5, # default
    'nu': 0.9, # vmafv4
}
