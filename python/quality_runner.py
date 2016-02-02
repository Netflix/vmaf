import re

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import time
import os
import multiprocessing
import subprocess
from python.config import PYTHON_ROOT
from tools import get_dir_without_last_slash, make_parent_dirs_if_nonexist
import numpy as np
import sys

class QualityRunner(object):

    def __init__(self,
                 assets,
                 logger,
                 log_file_dir= PYTHON_ROOT + "/../workspace/log_file_dir",
                 fifo_mode=True,
                 delete_workdir=True):

        self.assets = assets
        self.logger = logger
        self.log_file_dir = log_file_dir
        self.fifo_mode = fifo_mode
        self.delete_workdir = delete_workdir

    def run(self):

        if self.logger:
            self.logger.info(
                "For each asset, if {type} log has not been generated, "
                "run and generate {type} log file...".format(type=self.TYPE))

        # run generate_log_file on each asset
        map(self._run_and_generate_log_file_wrapper, self.assets)
        # from pathos.pp_map import pp_map
        # pp_map(self._run_and_generate_log_file_wrapper, self.assets)

        if self.logger:
            self.logger.info("Read {type} log file, get quality scores...".
                             format(type=self.TYPE))

        # collect result from each asset's log file
        results = map(self._read_result, self.assets)

        self.results = results

    def remove_logs(self):
        for asset in self.assets:
            self._remove_log(asset)
            self._remove_time_file(asset)

    def _run_and_generate_log_file(self, asset):

        log_file_path = self._get_log_file_path(asset)

        # touch (to start with a clean co)
        with open(log_file_path, 'wt'):
            pass

        # add runner type and version
        with open(log_file_path, 'at') as log_file:
            log_file.write("{type} VERSION {version}\n\n".format(
                type=self.TYPE, version=self.VERSION))

    def _run_and_generate_log_file_wrapper(self, asset):
        """
        Wraper around the essential function _run_and_generate_log_file, to
        do housekeeping work including 1) sanity check of asset, 2) skip run if
        log already exist, 3) creating fifo, 4) delete work file and dir, and
        5) log exec time.
        :param asset:
        :return:
        """

        # sanity check
        self._sanity_check(asset)

        log_file_path = self._get_log_file_path(asset)

        if os.path.isfile(log_file_path):
            if self.logger:
                self.logger.info(
                    '{type} log {log_file_path} exists. Skip {type} '
                    'run.'.format(type=self.TYPE,
                                  log_file_path=log_file_path))
        else:
            start_time = time.time()

            # remove workfiles if exist
            self._close_ref_workfile(asset)
            self._close_dis_workfile(asset)

            if self.fifo_mode:
                ref_p = multiprocessing.Process(target=self._open_ref_workfile,
                                                args=(asset, True))
                dis_p = multiprocessing.Process(target=self._open_dis_workfile,
                                                args=(asset, True))
                ref_p.start()
                dis_p.start()
            else:
                self._open_ref_workfile(asset, fifo_mode=False)
                self._open_dis_workfile(asset, fifo_mode=False)

            self._run_and_generate_log_file(asset)

            if self.delete_workdir:
                self._close_ref_workfile(asset)
                self._close_dis_workfile(asset)
                self._delete_workdir(asset)

            end_time = time.time()
            run_time = end_time - start_time

            if self.logger:
                self.logger.info(
                    "Run time: {sec:.2f} sec.".format(sec=run_time))

            self._write_time_file(asset, run_time)

    def _read_result(self, asset):

        result = {}

        # add quality scores
        result.update(self._get_quality_scores(asset))

        # add run time
        run_time = self._read_time_file(asset)
        time_key = "{type}_time".format(type=self.TYPE)
        result[time_key] = run_time

        # add dis video file bitrate (must be an entire file)
        dis_bitrate_kbps = asset.dis_bitrate_kbps_for_entire_file
        dis_bitrate_key = "dis_bitrate_kbps"
        result[dis_bitrate_key] = dis_bitrate_kbps

        return result

    def _sanity_check(self, asset):

        # 1) for now, quality width/height has to agree with ref/dis width/height
        assert asset.quality_width_height \
               == asset.ref_width_height \
               == asset.dis_width_height
        # 2) ...
        # 3) ...

    def _get_log_file_path(self, asset):
        return "{dir}/{type}/{str}".format(dir=self.log_file_dir,
                                           type=self.TYPE, str=str(asset))

    # ===== workfile =====

    def _open_ref_workfile(self, asset, fifo_mode):
        """
        For now, only works for YUV format -- all need is to copy from ref file
        to ref workfile
        :param asset:
        :param fifo_mode:
        :return:
        """
        src = asset.ref_path
        dst = asset.ref_workfile_path

        # if dst dir doesn't exist, create
        if self.logger:
            self.logger.info("Make parent directory for ref workfile {}".
                             format(dst))
        make_parent_dirs_if_nonexist(dst)

        # if fifo mode, mkfifo
        if fifo_mode:
            os.mkfifo(dst)

        # open ref file
        self._open_file(src, dst)

    def _open_dis_workfile(self, asset, fifo_mode):
        """
        For now, only works for YUV format -- all need is to copy from dis file
        to dis workfile
        :param asset:
        :param fifo_mode:
        :return:
        """
        src = asset.dis_path
        dst = asset.dis_workfile_path

        # if dst dir doesn't exist, create
        if self.logger:
            self.logger.info("Make parent directory for dis workfile {}".
                             format(dst))
        make_parent_dirs_if_nonexist(dst)

        # if fifo mode, mkfifo
        if fifo_mode:
            os.mkfifo(dst)

        # open dis file
        self._open_file(src, dst)

    def _delete_workdir(self, asset):
        ref_dir = get_dir_without_last_slash(asset.ref_workfile_path)
        dis_dir = get_dir_without_last_slash(asset.dis_workfile_path)

        assert ref_dir == dis_dir
        assert os.path.isdir(ref_dir)

        os.rmdir(ref_dir)


    def _open_file(self, src, dst):
        """
        For now, only works if source is YUV -- all needed is to copy
        :param src:
        :param dst:
        :return:
        """

        # NOTE: & is required for fifo mode !!!!
        cp_cmd = "cp {src} {dst} &". \
            format(src=src, dst=dst)
        if self.logger:
            self.logger.info(cp_cmd)
        subprocess.call(cp_cmd, shell=True)

    @staticmethod
    def _close_ref_workfile(asset):
        path = asset.ref_workfile_path
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _close_dis_workfile(asset):
        path = asset.dis_workfile_path
        if os.path.exists(path):
            os.remove(path)

    # ===== time file =====

    def _get_time_file_path(self, asset):
        return self._get_log_file_path(asset) + '.time'

    def _write_time_file(self, asset, time):
        time_file_path = self._get_time_file_path(asset)
        if os.path.exists(time_file_path):
            os.remove(time_file_path)
        with open(time_file_path, 'wt') as time_file:
            time_file.write(str(time))

    def _read_time_file(self, asset):
        time_file_path = self._get_time_file_path(asset)
        try:
            with open(time_file_path, 'rt') as time_file:
                time = map(float, time_file)[0]
        except RuntimeError as e:
            # if reading time file fails, shouldn't be catastrophic
            if self.logger:
                self.logger.error("Error reading time file: {e}".format(e=e))
            time = None
        return time

    def _remove_time_file(self, asset):
        time_file_path = self._get_time_file_path(asset)
        if os.path.exists(time_file_path):
            os.remove(time_file_path)


class VmafQualityRunner(QualityRunner):

    TYPE = 'VMAF'
    VERSION = '0.1'

    VMAF = PYTHON_ROOT + "/../feature/vmaf"
    SVM_MODEL_FILE = PYTHON_ROOT + "/../resource/model/model_V8a.model"
    FEAT_RESCALE = {'vif': (0.0, 1.0), 'adm': (0.4, 1.0),
                    'ansnr': (10.0, 50.0), 'motion': (0.0, 20.0)}

    sys.path.append(PYTHON_ROOT + "/../libsvm/python")
    import svmutil

    def _run_and_generate_log_file(self, asset):

        super(VmafQualityRunner, self)._run_and_generate_log_file(asset)

        log_file_path = self._get_log_file_path(asset)

        # if parent dir doesn't exist, create
        make_parent_dirs_if_nonexist(log_file_path)

        # run VMAF command line to extract features, 'APPEND' result (since
        # super method already does something
        quality_width, quality_height = asset.quality_width_height
        vmaf_cmd = """
        {vmaf} vif {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path};
        {vmaf} adm {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path};
        {vmaf} ansnr {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path};
        {vmaf} motion {yuv_type} {ref_path} {dis_path} {w} {h} >> {log_file_path};
        """.format(
            vmaf=self.VMAF,
            yuv_type=asset.yuv_type,
            ref_path=asset.ref_path,
            dis_path=asset.dis_path,
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
        scores = []
        for vif, adm, ansnr, motion in zip(feature_result['vif_scores'],
                                       feature_result['adm_scores'],
                                       feature_result['ansnr_scores'],
                                       feature_result['motion_scores']):
            xs = [[
                self._rescale(vif, self.FEAT_RESCALE['vif'][0],
                             self.FEAT_RESCALE['vif'][1]),
                self._rescale(adm, self.FEAT_RESCALE['adm'][0],
                             self.FEAT_RESCALE['adm'][1]),
                self._rescale(ansnr, self.FEAT_RESCALE['ansnr'][0],
                             self.FEAT_RESCALE['ansnr'][1]),
                self._rescale(motion, self.FEAT_RESCALE['motion'][0],
                             self.FEAT_RESCALE['motion'][1])
            ]]

            score = self.svmutil.svm_predict([0], xs, model)[0][0]

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

            scores.append(score)

        # append final VMAF scores to log file
        with open(log_file_path, 'at') as log_file:
            for idx, score in enumerate(scores):
                log_file.write("vmaf: {idx} {score}\n".format(idx=idx,
                                                              score=score))

    @staticmethod
    def _rescale(val, mi, ma):
            val = np.double(val)
            val = (val - mi)/(ma - mi)
            if val < 0:
                val = 0
            elif val > 1:
                val = 1
            val = np.double(val)
            return val

    def _get_quality_scores(self, asset):

        feat_result = self._get_feature_scores(asset)

        result = {}

        # individual feature scores
        result.update(feat_result)

        # add aggregate feature scores
        result['vif_score'] = np.mean(feat_result['vif_scores'])
        result['adm_score'] = np.mean(feat_result['adm_scores'])
        result['ansnr_score'] = np.mean(feat_result['ansnr_scores'])
        result['motion_score'] = np.mean(feat_result['motion_scores'])

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
        result[self.TYPE + "_score"] = np.mean(vmaf_scores)

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

        assert len(vif_scores) == len(adm_scores) == len(ansnr_scores) == len(motion_scores)

        feat_result = {}

        feat_result['vif_scores'] = vif_scores
        feat_result['adm_scores'] = adm_scores
        feat_result['ansnr_scores'] = ansnr_scores
        feat_result['motion_scores'] = motion_scores

        return feat_result

    def _remove_log(self, asset):
        log_file_path = self._get_log_file_path(asset)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
