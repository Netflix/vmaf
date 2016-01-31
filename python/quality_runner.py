__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import time
import os
from multiprocessing import Process

class QualityRunner(object):

    def __init__(self, assets, logger, log_file_dir='../workspace/log_file_dir',
                 pipe_mode=True):
        self.assets = assets
        self.logger = logger
        self.log_file_dir = log_file_dir
        self.pipe_mode = pipe_mode

    def run(self):

        if self.logger:
            self.logger.info(
                "For each asset, if {type} log has not been generated, "
                "run and generate {type} log file...}").format(type=self.TYPE)

        for asset in self.assets:

            log_file_path = self._get_log_file_path(asset)

            if os.path.isfile(log_file_path):
                if self.logger:
                    self.logger.info(
                        '{type} log {log_file_path} exists. Skip {type} '
                        'run.'.format(type=self.TYPE,
                                              log_file_path=log_file_path))
                continue

            start_time = time.time()

            # remove pipes if exists
            self._close_ref_workfile(asset)
            self._close_dis_workfile(asset)

            if self.pipe_mode:
                Process(target=self._open_ref_workfile, args=(asset, True)).start()
                Process(target=self._open_dis_workfile, args=(asset, True)).start()
            else:
                self._open_ref_workfile(asset, pipe_mode=False)
                self._open_dis_workfile(asset, pipe_mode=False)

            self._run_and_generate_log_file(asset)

            self._close_ref_workfile(asset)
            self._close_dis_workfile(asset)

            end_time = time.time()
            run_time = end_time - start_time

            if self.logger:
                self.logger.info("Run time: {sec:.2f} sec.".format(sec=run_time))

            self._write_time_file(asset, run_time)

        if self.logger:
            self.logger.info("Read {type} log file, get quality scores...".
                             format(type=self.TYPE))
        results = []
        for asset in self.assets:

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

            results.append(result)

        self.results = results

    def remove_logs(self):
        for asset in self.assets:
            self._remove_log(asset)
            self._remove_time_file(asset)

    def _get_log_file_path(self, asset):
        return "{dir}/{type}/{str}".format(dir=self.log_file_dir,
                                           type=self.TYPE, str=str(asset))

    @staticmethod
    def _open_ref_workfile(asset, pipe_mode):
        pass

    @staticmethod
    def _open_dis_workfile(asset, pipe_mode):
        pass

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

    @classmethod
    def _get_time_file_path(cls, asset):
        return cls._get_log_file_path(asset) + '.time'

    @classmethod
    def _write_time_file(cls, asset, time):
        time_file_path = cls._get_time_file_path(asset)
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

    @classmethod
    def _remove_time_file(cls, asset):
        time_file_path = cls._get_time_file_path(asset)
        if os.path.exists(time_file_path):
            os.remove(time_file_path)

class VmafQualityRunner(QualityRunner):

    TYPE = 'VMAF'
    VERSION = '0.1'
