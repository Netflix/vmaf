model_type = "RANDOMFOREST"
model_param_dict = {

    'norm_type':'none', # default - wont' matter for random forest

    'n_estimators':10, # default

    'criterion':'mse', # default

    'max_depth':None, # default

    # 'random_state':None, # default
    'random_state':0, # make result deterministic

}
