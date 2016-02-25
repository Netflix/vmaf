model_type = "LIBSVMNUSVR"
model_param_dict = {

    # 'norm_type':'none', # default
    'norm_type':'clip_0to1', # selected
    # 'norm_type':'clip_minus1to1',
    # 'norm_type':'normalize',

    # 'gamma':0.0, # default
    'gamma':0.85, # selected

    'C':1.0, # default

    'nu':0.5, # default

    'cache_size':200 # default

}
