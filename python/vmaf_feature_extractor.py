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

    ATOM_FEATURES = {'vif', 'adm', 'ansnr', 'motion'}

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

        vif_scores = []
        adm_scores = []
        ansnr_scores = []
        motion_scores = []
        vif_idx = 0
        adm_idx = 0
        ansnr_idx = 0
        motion_idx = 0
        with open(log_file_path, 'rt') as log_file:
            for line in log_file.readlines():
                mo_vif = re.match(r"vif: ([0-9]+) ([0-9.]+)", line)
                if mo_vif:
                    cur_vif_idx = int(mo_vif.group(1))
                    assert cur_vif_idx == vif_idx
                    vif_scores.append(float(mo_vif.group(2)))
                    vif_idx += 1
                else:
                    mo_adm = re.match(r"adm: ([0-9]+) ([0-9.]+)", line)
                    if mo_adm:
                        cur_adm_idx = int(mo_adm.group(1))
                        assert cur_adm_idx == adm_idx
                        adm_scores.append(float(mo_adm.group(2)))
                        adm_idx += 1
                    else:
                        mo_ansnr = re.match(r"ansnr: ([0-9]+) ([0-9.-]+)", line)
                        if mo_ansnr:
                            cur_ansnr_idx = int(mo_ansnr.group(1))
                            assert cur_ansnr_idx == ansnr_idx
                            ansnr_scores.append(float(mo_ansnr.group(2)))
                            ansnr_idx += 1
                        else:
                            mo_motion = re.match(r"motion: ([0-9]+) ([0-9.-]+)", line)
                            if mo_motion:
                                cur_motion_idx = int(mo_motion.group(1))
                                assert cur_motion_idx == motion_idx
                                motion_scores.append(float(mo_motion.group(2)))
                                motion_idx += 1

        assert len(vif_scores) == len(adm_scores) == \
               len(ansnr_scores) == len(motion_scores)

        feature_result = {}

        feature_result[self.TYPE + '_vif_scores'] = vif_scores
        feature_result[self.TYPE + '_adm_scores'] = adm_scores
        feature_result[self.TYPE + '_ansnr_scores'] = ansnr_scores
        feature_result[self.TYPE + '_motion_scores'] = motion_scores

        return feature_result

    def _remove_log(self, asset):
        log_file_path = self._get_log_file_path(asset)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)