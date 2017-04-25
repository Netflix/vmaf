import os
import subprocess
import re
import sys

import numpy as np

from vmaf.core.asset import Asset
from vmaf.core.quality_runner import QualityRunner
from vmaf.tools.misc import make_absolute_path, run_process
from vmaf.config import VmafExternalConfig
from vmaf.config import VmafConfig


__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class VqmAsset(Asset):

    @property
    def ref_workfile_path(self):
        return super(VqmAsset, self).ref_workfile_path + '.avi'

    @property
    def dis_workfile_path(self):
        return super(VqmAsset, self).dis_workfile_path + '.avi'

    @classmethod
    def from_asset(cls, asset):
        vqm_asset = cls(dataset=asset.dataset,
                   content_id=asset.content_id,
                   asset_id=asset.asset_id,
                   ref_path=asset.ref_path,
                   dis_path=asset.dis_path,
                   asset_dict=asset.asset_dict,
                   )
        vqm_asset.workdir = asset.workdir
        return vqm_asset

class VqmQualityRunner(QualityRunner):

    def __init__(self,
                 assets,
                 logger,
                 fifo_mode=True,
                 delete_workdir=True,
                 result_store=None,
                 optional_dict=None,
                 optional_dict2=None,
                 ):
        # override Executor.__init__ by replacing assets of type Asset with
        # VqmAsset, which will yield different ref/dis_workfile_path

        assert fifo_mode is False, \
            'VqmQualityRunner calls matlab, which cannot use fifo mode.'

        super(VqmQualityRunner, self).__init__(assets, logger,
                                               fifo_mode, delete_workdir,
                                               result_store,
                                               optional_dict, optional_dict2)
        self.assets = map(lambda x: VqmAsset.from_asset(x), self.assets)

    @classmethod
    def _set_asset_use_path_as_workpath(cls, asset):
        # Override Executor._set_asset_use_path_as_workpath(asset).
        # For VqmQualityRunner, never use ref/dis_path as workpath, for
        # the input encodes needs to be format converted first
        asset.use_path_as_workpath = False

    def _open_ref_workfile(self, asset, fifo_mode):
        # override Executor._open_ref_workfile
        # unlike others, VQM quality runners (general, vfd)
        # require input of format AVI (wrapping raw YUV). Instead copying
        # file, call ffmpeg command

        # only need to open ref workfile if the path is different from ref path
        assert asset.use_path_as_workpath == False \
               and asset.ref_path != asset.ref_workfile_path

        # if fifo mode, mkfifo
        if fifo_mode:
            os.mkfifo(asset.ref_workfile_path)

        quality_width, quality_height = asset.quality_width_height

        width, height = asset.ref_width_height
        src_fmt_cmd = self._get_yuv_src_fmt_cmd(asset, height, width)
        if asset.fps:
            src_fmt_cmd += ' -r {}'.format(int(asset.fps))

        # VQM can only take input file of yuyv422 wrapped in avi
        ffmpeg_cmd = '{ffmpeg} {src_fmt_cmd} -i {src} -an -vsync 0 ' \
                     '-vcodec rawvideo -pix_fmt yuyv422 -vf scale={width}x{height} -f avi ' \
                     '-sws_flags bilinear -y {dst}'.format(
            ffmpeg=VmafExternalConfig.get_and_assert_ffmpeg(), src=asset.ref_path, dst=asset.ref_workfile_path,
            width=quality_width, height=quality_height,
            src_fmt_cmd=src_fmt_cmd,
            # yuv_fmt=asset.yuv_type
        )

        if self.logger:
            self.logger.info(ffmpeg_cmd)
        run_process(ffmpeg_cmd, shell=True)

    def _open_dis_workfile(self, asset, fifo_mode):
        # override Executor._open_dis_workfile
        # unlike others, VQM quality runners (general, vfd)
        # require input of format AVI (wrapping raw YUV). Instead copying
        # file, call ffmpeg command

        # only need to open dis workfile if the path is different from dis path
        assert asset.use_path_as_workpath == False \
               and asset.dis_path != asset.dis_workfile_path

        # if fifo mode, mkfifo
        if fifo_mode:
            os.mkfifo(asset.dis_workfile_path)

        quality_width, quality_height = asset.quality_width_height

        width, height = asset.dis_width_height
        src_fmt_cmd = self._get_yuv_src_fmt_cmd(asset, height, width)
        if asset.fps:
            src_fmt_cmd += ' -r {}'.format(int(asset.fps))

        # VQM can only take input file of yuyv422 wrapped in avi
        ffmpeg_cmd = '{ffmpeg} {src_fmt_cmd} -i {src} -an -vsync 0 ' \
                     '-vcodec rawvideo -pix_fmt yuyv422 -vf scale={width}x{height} -f avi ' \
                     '-sws_flags bilinear -y {dst}'.format(
            ffmpeg=VmafExternalConfig.get_and_assert_ffmpeg(), src=asset.dis_path, dst=asset.dis_workfile_path,
            width=quality_width, height=quality_height,
            src_fmt_cmd=src_fmt_cmd,
            # yuv_fmt=asset.yuv_type
        )

        if self.logger:
            self.logger.info(ffmpeg_cmd)
        run_process(ffmpeg_cmd, shell=True)

    @classmethod
    def _assert_an_asset(cls, asset):
        # override Executor._assert_an_asset

        super(VqmQualityRunner, cls)._assert_an_asset(asset)

        assert asset.yuv_type != 'notyuv', 'need to modify and test in notyuv case.'

        assert asset.ref_duration_sec is None or asset.ref_duration_sec >= 4.0, \
            "Ref video too short. CVQM requires at least 4-seconds of video to run."
        assert asset.ref_duration_sec is None or asset.dis_duration_sec >= 4.0, \
            "Dis video too short. CVQM requires at least 4-seconds of video to run."

    @classmethod
    def _post_process_result(cls, result):
        # override Executor._post_process_result

        result = super(VqmQualityRunner, cls)._post_process_result(result)

        scores_key = cls.get_scores_key()
        result.result_dict[scores_key] = list(1.0 - np.array(result.result_dict[scores_key]))
        return result

    def _get_vqm_matlab_workspace(self):
        # depends on Mac or Linux, switch to different dir
        if sys.platform == 'linux' or sys.platform == 'linux2':
	    os.environ['MCR_CACHE_ROOT'] = "/tmp"
            return VmafConfig.root_path('matlab', 'vqm', 'VQM_Linux')
        elif sys.platform == 'darwin':
            os.environ['MCR_CACHE_ROOT'] = "/private/tmp"
	    return VmafConfig.root_path('matlab', 'vqm', 'VQM_Mac')
        else:
            raise AssertionError("System is {}, which is not supported by VQM".format(sys.platform))

class VqmGeneralQualityRunner(VqmQualityRunner):

    TYPE = 'VQM_General'

    # VERSION = '1.0'
    VERSION = '1.1' # call MATLAB runtime instead of MATLAB

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        ref_workfile_path = asset.ref_workfile_path
        dis_workfile_path = asset.dis_workfile_path

        current_dir = os.getcwd() + '/'

        ref_workfile_path = make_absolute_path(ref_workfile_path, current_dir)
        dis_workfile_path = make_absolute_path(dis_workfile_path, current_dir)

        vqm_matlab_workspace = self._get_vqm_matlab_workspace()

        vqm_cmd = """{workspace}/run_cvqm.sh {matlab_runtime} '{original_file}' '{processed_file}' 'progressive' 'none' 'general'""".format(
            workspace=vqm_matlab_workspace,
            matlab_runtime=VmafExternalConfig.get_and_assert_matlab_runtime(),
            original_file=ref_workfile_path,
            processed_file=dis_workfile_path,
        )
        if self.logger:
            self.logger.info(vqm_cmd)

        run_process(vqm_cmd, shell=True)

        vqm_calibration_filepath = dis_workfile_path + '_calibration.txt'
        vqm_model_filepath = dis_workfile_path + '_model.txt'

        log_file_path = self._get_log_file_path(asset)
        log_file_path = make_absolute_path(log_file_path, current_dir)

        mv_cmd = '''mv {src} {dst}'''.format(src=vqm_model_filepath, dst=log_file_path)
        if self.logger:
            self.logger.info(mv_cmd)
        run_process(mv_cmd, shell=True)

        rm_cmd = '''rm {}'''.format(vqm_calibration_filepath)
        if self.logger:
            self.logger.info(rm_cmd)
        run_process(rm_cmd, shell=True)

    def _get_quality_scores(self, asset):
        # routine to read the quality scores from the log file, and return
        # the scores in a dictionary format.

        log_file_path = self._get_log_file_path(asset)
        vqm_scores = []
        with open(log_file_path, 'rt') as log_file:
            for line in log_file.readlines():
                mo = re.match(r"([0-9.]+) general", line)
                if mo:
                    vqm_scores.append(float(mo.group(1)))
        assert len(vqm_scores) == 1

        scores_key = self.get_scores_key()
        quality_result = {
            scores_key:vqm_scores
        }
        return quality_result

class VqmVfdQualityRunner(VqmQualityRunner):

    TYPE = 'VQM_VFD'

    # VERSION = '1.0'
    VERSION = '1.1' # call MATLAB runtime instead of MATLAB

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        ref_workfile_path = asset.ref_workfile_path
        dis_workfile_path = asset.dis_workfile_path

        current_dir = os.getcwd() + '/'

        ref_workfile_path = make_absolute_path(ref_workfile_path, current_dir)
        dis_workfile_path = make_absolute_path(dis_workfile_path, current_dir)

        vqm_matlab_workspace = self._get_vqm_matlab_workspace()

        vqm_cmd = """{workspace}/run_cvqm.sh {matlab_runtime} '{original_file}' '{processed_file}' 'progressive' 'none' 'vqm_vfd'""".format(
            workspace=vqm_matlab_workspace,
            matlab_runtime=VmafExternalConfig.get_and_assert_matlab_runtime(),
            original_file=ref_workfile_path,
            processed_file=dis_workfile_path,
        )
        if self.logger:
            self.logger.info(vqm_cmd)

        run_process(vqm_cmd, shell=True)

        vqm_calibration_filepath = dis_workfile_path + '_calibration.txt'
        vqm_model_filepath = dis_workfile_path + '_model.txt'
        vqm_csv_filepath = dis_workfile_path + '_vfd.csv'

        log_file_path = self._get_log_file_path(asset)
        log_file_path = make_absolute_path(log_file_path, current_dir)

        mv_cmd = '''mv {src} {dst}'''.format(src=vqm_model_filepath, dst=log_file_path)
        if self.logger:
            self.logger.info(mv_cmd)
        run_process(mv_cmd, shell=True)

        rm_cmd = '''rm {}'''.format(vqm_calibration_filepath)
        if self.logger:
            self.logger.info(rm_cmd)
        run_process(rm_cmd, shell=True)

        rm_csv_cmd = '''rm {}'''.format(vqm_csv_filepath)
        if self.logger:
            self.logger.info(rm_csv_cmd)
        run_process(rm_csv_cmd, shell=True)

    def _get_quality_scores(self, asset):
        # routine to read the quality scores from the log file, and return
        # the scores in a dictionary format.

        log_file_path = self._get_log_file_path(asset)
        vqm_scores = []
        with open(log_file_path, 'rt') as log_file:
            for line in log_file.readlines():
                mo = re.match(r"([0-9.]+) vqm_vfd", line)
                if mo:
                    vqm_scores.append(float(mo.group(1)))
        assert len(vqm_scores) == 1

        scores_key = self.get_scores_key()
        quality_result = {
            scores_key:vqm_scores
        }
        return quality_result

