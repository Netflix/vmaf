feature_dict = {

    # use selected features from VmafFeatureExtractor
    'VMAF_feature': ['vif', 'adm', 'motion', 'ansnr'],

    'Moment_feature':['dis1st'], # use distorted video's 1st moment
}

feature_optional_dict = {
    'VMAF_feature': {'adm_ref_display_height': 108000},
}
