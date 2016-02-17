__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import re
import subprocess
import config
from feature_extractor import FeatureExtractor

class VmafFeatureExtractor(FeatureExtractor):

    TYPE = "VMAF_feature"
    VERSION = '0.1'

    ATOM_FEATURES = ['vif', 'adm', 'ansnr', 'motion']

    VMAF_FEATURE = config.ROOT + "/feature/vmaf"

    def _run_and_generate_log_file(self, asset):

        super(VmafFeatureExtractor, self)._run_and_generate_log_file(asset)

        log_file_path = self._get_log_file_path(asset)

        # run VMAF command line to extract features, 'APPEND' result (since
        # super method already does something
        quality_width, quality_height = asset.quality_width_height
        vmaf_feature_cmd = "{vmaf} all {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path}" \
        .format(
            vmaf=self.VMAF_FEATURE,
            yuv_type=asset.yuv_type,
            ref_path=asset.ref_workfile_path,
            dis_path=asset.dis_workfile_path,
            w=quality_width,
            h=quality_height,
            log_file_path=log_file_path,
        )

        if self.logger:
            self.logger.info(vmaf_feature_cmd)

        subprocess.call(vmaf_feature_cmd, shell=True)

    def _get_feature_scores(self, asset):
        log_file_path = self._get_log_file_path(asset)

        atom_feature_scores_dict = {}
        atom_feature_idx_dict = {}
        for atom_feature in self.ATOM_FEATURES:
            atom_feature_scores_dict[atom_feature] = []
            atom_feature_idx_dict[atom_feature] = 0

        with open(log_file_path, 'rt') as log_file:
            for line in log_file.readlines():
                for atom_feature in self.ATOM_FEATURES:
                    re_template = "{af}: ([0-9]+) ([0-9.-]+)".format(af=atom_feature)
                    mo = re.match(re_template, line)
                    if mo:
                        cur_idx = int(mo.group(1))
                        assert cur_idx == atom_feature_idx_dict[atom_feature]
                        atom_feature_scores_dict[atom_feature].append(float(mo.group(2)))
                        atom_feature_idx_dict[atom_feature] += 1
                        continue

        len_score = len(atom_feature_scores_dict[self.ATOM_FEATURES[0]])
        for atom_feature in self.ATOM_FEATURES[1:]:
            assert len_score == len(atom_feature_scores_dict[atom_feature])

        feature_result = {}

        for atom_feature in self.ATOM_FEATURES:
            scores_key = self._get_scores_key(atom_feature)
            feature_result[scores_key] = atom_feature_scores_dict[atom_feature]

        return feature_result

    def _remove_log(self, asset):
        log_file_path = self._get_log_file_path(asset)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)