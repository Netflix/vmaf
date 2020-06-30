import copy
import os
import pickle
import shutil
import sys

# from vmaf.config import VmafConfig

assert sys.version_info[0] == 2, 'Must to use py2 to generate the output pickle file.'

def convert_vmaf_model_to_vmaf_no_enhn_gain_model(vmaf_model_path, output_vmaf_neg_model_path):
    with open(vmaf_model_path, 'rb') as file:
        vmaf_model = pickle.load(file,
                                 # encoding='latin1',
                                 )
        vmaf_neg_model = copy.deepcopy(vmaf_model)
        vmaf_neg_model['model_dict']['enhn_gain'] = dict()
        vmaf_neg_model['model_dict']['enhn_gain']['vif_enhn_gain_limit'] = 1.0
        vmaf_neg_model['model_dict']['enhn_gain']['adm_enhn_gain_limit'] = 1.0
    if not os.path.exists(os.path.dirname(output_vmaf_neg_model_path)):
        os.makedirs(os.path.dirname(output_vmaf_neg_model_path))
    with open(output_vmaf_neg_model_path, 'wb') as output_file:
        pickle.dump(vmaf_neg_model, output_file,
                    protocol=1
                    )

    vmaf_svm_model_path = vmaf_model_path + '.model'
    output_vmaf_neg_svm_model_path = output_vmaf_neg_model_path + '.model'
    shutil.copyfile(vmaf_svm_model_path, output_vmaf_neg_svm_model_path)


convert_vmaf_model_to_vmaf_no_enhn_gain_model(
    # VmafConfig.model_path('vmaf_v0.6.1.pkl'),
    './model/vmaf_v0.6.1.pkl',
    # VmafConfig.workspace_path('model', 'vmaf_v0.6.1neg.pkl')
    './model/vmaf_v0.6.1neg.pkl',
)

print('Done.')
