import copy
import os
import pickle
import shutil
import sys

# from vmaf.config import VmafConfig


def convert_vmaf_model_to_vmaf_no_enhn_gain_model(vmaf_model_path, output_vmaf_neg_model_path):
    with open(vmaf_model_path, 'rb') as file:
        vmaf_model = pickle.load(file,
                                 # encoding='latin1',
                                 )
        vmaf_neg_model = copy.deepcopy(vmaf_model)
        vmaf_neg_model['model_dict']['feature_opts_dicts'] = [
            {'adm_enhn_gain_limit': 1.0},  # 'VMAF_feature_adm2_score'
            {},                            # 'VMAF_feature_motion2_score'
            {'vif_enhn_gain_limit': 1.0},  # 'VMAF_feature_vif_scale0_score'
            {'vif_enhn_gain_limit': 1.0},  # 'VMAF_feature_vif_scale1_score'
            {'vif_enhn_gain_limit': 1.0},  # 'VMAF_feature_vif_scale2_score'
            {'vif_enhn_gain_limit': 1.0},  # 'VMAF_feature_vif_scale3_score'
        ]
   
    os.makedirs(os.path.dirname(output_vmaf_neg_model_path), exist_ok=True)
    with open(output_vmaf_neg_model_path, 'wb') as output_file:
        pickle.dump(vmaf_neg_model, output_file,
                    protocol=1
                    )

    vmaf_svm_model_path = vmaf_model_path + '.model'
    output_vmaf_neg_svm_model_path = output_vmaf_neg_model_path + '.model'
    shutil.copyfile(vmaf_svm_model_path, output_vmaf_neg_svm_model_path)


if not sys.version_info[0] == 2:
    print('warning: running {} skipped - must to use py2 to generate the output pickle file.'.format(
        os.path.basename(__file__)))
else:
    convert_vmaf_model_to_vmaf_no_enhn_gain_model(
        # VmafConfig.model_path('vmaf_float_v0.6.1.pkl'),
        './model/vmaf_float_v0.6.1.pkl',
        # VmafConfig.workspace_path('model', 'vmaf_float_v0.6.1neg.pkl')
        './model/vmaf_float_v0.6.1neg.pkl',
    )
    print('Done.')
