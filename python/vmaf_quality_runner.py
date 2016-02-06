from functools import partial

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import re
import subprocess
import sys
import config
from quality_runner import QualityRunner

class VmafQualityRunner(QualityRunner):

    TYPE = 'VMAF'
    VERSION = '0.1'

    VMAF = config.ROOT + "/feature/vmaf"
    SVM_MODEL_FILE = config.ROOT + "/resource/model/model_V8a.model"
    FEAT_RESCALE = {'vif': (0.0, 1.0), 'adm': (0.4, 1.0),
                    'ansnr': (10.0, 50.0), 'motion': (0.0, 20.0)}

    sys.path.append(config.ROOT + "/libsvm/python")
    import svmutil

    def _asserts(self):
        super(VmafQualityRunner, self)._asserts()

        # for now, VMAF c code don't work in fifo mode yet
        # TODO: fix fifo mode
        assert self.fifo_mode is False, \
            "For now, VmafQualityRunner do not support fifo mode."

    def _run_and_generate_log_file(self, asset):

        super(VmafQualityRunner, self)._run_and_generate_log_file(asset)

        log_file_path = self._get_log_file_path(asset)

        # run VMAF command line to extract features, 'APPEND' result (since
        # super method already does something
        quality_width, quality_height = asset.quality_width_height
        vmaf_cmd = "{vmaf} all {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path}" \
        .format(
            vmaf=self.VMAF,
            yuv_type=asset.yuv_type,
            ref_path=asset.ref_workfile_path,
            dis_path=asset.dis_workfile_path,
            w=quality_width,
            h=quality_height,
            log_file_path=log_file_path,
        )

        if self.logger:
            self.logger.info(vmaf_cmd)

        subprocess.call(vmaf_cmd, shell=True)

        # read feature from log file, run regressor prediction
        feature_result = self._get_feature_scores(asset)
        model = self.svmutil.svm_load_model(self.SVM_MODEL_FILE)

        scaled_vif_scores = self._rescale(
            feature_result[self.TYPE + '_vif_scores'], self.FEAT_RESCALE['vif'])
        scaled_adm_scores = self._rescale(
            feature_result[self.TYPE + '_adm_scores'], self.FEAT_RESCALE['adm'])
        scaled_ansnr_scores = self._rescale(
            feature_result[self.TYPE + '_ansnr_scores'], self.FEAT_RESCALE['ansnr'])
        scaled_motion_scores = self._rescale(
            feature_result[self.TYPE + '_motion_scores'], self.FEAT_RESCALE['motion'])

        scores = []
        for vif, adm, ansnr, motion in zip(scaled_vif_scores,
                                           scaled_adm_scores,
                                           scaled_ansnr_scores,
                                           scaled_motion_scores):
            xs = [[vif, adm, ansnr, motion]]
            score = self.svmutil.svm_predict([0], xs, model)[0][0]
            score = self._post_correction(motion, score)
            scores.append(score)

        # append final VMAF scores to log file
        with open(log_file_path, 'at') as log_file:
            for idx, score in enumerate(scores):
                log_file.write("vmaf: {idx} {score}\n".format(idx=idx,
                                                              score=score))

    def _post_correction(self, motion, score):
        # post-SVM correction
        if motion > 12.0:
            val = motion
            if val > 20.0:
                val = 20
            score *= ((val - 12) * 0.015 + 1)
        if score > 100.0:
            score = 100.0
        elif score < 0.0:
            score = 0.0
        return score

    @classmethod
    def _rescale(cls, vals, lower_upper_bound):
        # import numpy as np
        # lower_bound, upper_bound = lower_upper_bound
        # vals = np.double(vals)
        # vals = np.clip(vals, lower_bound, upper_bound)
        # vals = (vals - lower_bound)/(upper_bound - lower_bound)
        # return vals
        # avoid dependency on numpy here:
        _rescale_scaler_partial = \
            partial(cls._rescale_scalar,
                    lower_upper_bound=lower_upper_bound)
        return map(_rescale_scaler_partial, vals)

    @staticmethod
    def _rescale_scalar(val, lower_upper_bound):
        """
        Scalar version of _rescale.
        :param val:
        :param lower_upper_bound:
        :return:
        """
        lower_bound, upper_bound = lower_upper_bound
        val = val if val > lower_bound else lower_bound
        val = val if val < upper_bound else upper_bound
        val = (float(val) - lower_bound) / (upper_bound - lower_bound)
        return val

    def _get_quality_scores(self, asset):

        feat_result = self._get_feature_scores(asset)

        result = {}

        # individual feature scores
        result.update(feat_result)

        # read VMAF scores
        log_file_path = self._get_log_file_path(asset)
        vmaf_scores = []
        vmaf_idx = 0
        with open(log_file_path, 'rt') as log_file:
            for line in log_file.readlines():
                mo_vmaf = re.match(r"vmaf: ([0-9]+) ([0-9.]+)", line)
                if mo_vmaf:
                    cur_vmaf_idx = int(mo_vmaf.group(1))
                    assert cur_vmaf_idx == vmaf_idx
                    vmaf_scores.append(float(mo_vmaf.group(2)))
                    vmaf_idx += 1

        # add VMAF scores
        result[self.TYPE + "_scores"] = vmaf_scores

        return result

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

        feat_result = {}

        feat_result[self.TYPE + '_vif_scores'] = vif_scores
        feat_result[self.TYPE + '_adm_scores'] = adm_scores
        feat_result[self.TYPE + '_ansnr_scores'] = ansnr_scores
        feat_result[self.TYPE + '_motion_scores'] = motion_scores

        return feat_result

    def _remove_log(self, asset):
        log_file_path = self._get_log_file_path(asset)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)