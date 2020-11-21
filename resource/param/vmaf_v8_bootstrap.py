feature_dict = {

   'VMAF_feature': ['vif_scale0', 'vif_scale1', 'vif_scale2', 'vif_scale3',
                    'adm2', 'motion2',],

}

model_type = "BOOTSTRAP_LIBSVMNUSVR"
model_param_dict = {

    # ==== preprocess: normalize each feature ==== #
    # 'norm_type': 'none', # default: do nothing
    'norm_type': 'clip_0to1', # rescale to within [0, 1]
    # 'norm_type': 'clip_minus1to1', # rescale to within [-1, 1]
    # 'norm_type': 'normalize', # rescale to mean zero and std one

    # ==== postprocess: clip final quality score ==== #
    # 'score_clip': None, # default: do nothing
    'score_clip': [0.0, 100.0], # clip to within [0, 100]

    # ==== postprocess: transform final quality score ==== #
    'score_transform': {'p0':1.70674692, 'p1':1.72643844, 'p2':-0.00705305, 'out_gte_in':'true'},  # laptop vs. mobile transform

    # ==== libsvmnusvr parameters ==== #

    'gamma': 0.04,
    'C': 4.0,
    'nu': 0.9,

    # ==== bootstrap parameters ==== #

    'num_models': 101,  # this leads to 100 bootstrapped models being trained
}
