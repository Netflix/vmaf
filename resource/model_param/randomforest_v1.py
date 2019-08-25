model_type = "RANDOMFOREST"
model_param_dict = {

    # ==== preprocess: normalize each feature ==== #
    'norm_type':'none', # default: do nothing - won't matter to random forest

    # ==== postprocess: clip final quality score
    # 'score_clip':None, # default: do nothing
    'score_clip':[0.0, 100.0], # clip to within [0, 100]

    # ==== randomforest parameters ==== #

    'n_estimators':10, # default

    'criterion':'mse', # default

    'max_depth':None, # default

    # 'random_state':None, # default: random
    'random_state':0, # make result deterministic

}
