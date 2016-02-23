from result_store import ResultStore

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import re
import time
import os
import multiprocessing
import subprocess
import config
from tools import get_dir_without_last_slash, make_parent_dirs_if_nonexist

class QualityRunner(object):

    def __init__(self,
                 assets,
                 logger,
                 log_file_dir= config.ROOT + "/workspace/log_file_dir",
                 fifo_mode=True,
                 delete_workdir=True):

        self.assets = assets
        self.logger = logger
        self.log_file_dir = log_file_dir
        self.fifo_mode = fifo_mode
        self.delete_workdir = delete_workdir
        self.results = []

        self._asserts()

    def _asserts(self):
        assert hasattr(self, "TYPE")
        assert hasattr(self, "VERSION")

        assert re.match(r"[a-zA-Z0-9_]+", self.TYPE), \
            "TYPE can only contains alphabets, numbers and _."

        list_contentid_assetid_pair = \
            map(lambda asset: (asset.content_id, asset.asset_id), self.assets)
        assert len(list_contentid_assetid_pair) == \
               len(set(list_contentid_assetid_pair)), \
            "Pair of content_id and asset_id must be unique for each asset."

    def run(self):

        if self.logger:
            self.logger.info(
                "For each asset, if {type} log has not been generated, "
                "run and generate {type} log file...".format(type=self.TYPE))

        # run generate_log_file on each asset
        map(self._run_and_generate_log_file_wrapper, self.assets)

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

        # if parent dir doesn't exist, create
        make_parent_dirs_if_nonexist(log_file_path)

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
        do housekeeping work including 1) asserts of asset, 2) skip run if
        log already exist, 3) creating fifo, 4) delete work file and dir, and
        5) log exec time.
        :param asset:
        :return:
        """

        # asserts
        self._asserts_asset(asset)

        log_file_path = self._get_log_file_path(asset)

        if os.path.isfile(log_file_path):
            if self.logger:
                self.logger.info(
                    '{type} log {log_file_path} exists. Skip {type} '
                    'run.'.format(type=self.TYPE,
                                  log_file_path=log_file_path))
        else:
            start_time = time.time()

            # remove workfiles if exist (do early here to avoid race condition
            # when ref path and dis path have some overlap)
            self._close_ref_workfile(asset)
            self._close_dis_workfile(asset)

            make_parent_dirs_if_nonexist(asset.ref_workfile_path)
            make_parent_dirs_if_nonexist(asset.dis_workfile_path)

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

                ref_dir = get_dir_without_last_slash(asset.ref_workfile_path)
                dis_dir = get_dir_without_last_slash(asset.dis_workfile_path)
                os.rmdir(ref_dir)
                try:
                    os.rmdir(dis_dir)
                except OSError as e:
                    if e.errno == 2: # [Errno 2] No such file or directory
                        # already removed by os.rmdir(ref_dir)
                        pass

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

        return ResultStore(asset, result)

    def _asserts_asset(self, asset):

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

        # if fifo mode, mkfifo
        if fifo_mode:
            os.mkfifo(dst)

        # open dis file
        self._open_file(src, dst)

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


def run_quality_runners_in_parallel(runner_class,
                                    assets,
                                    log_file_dir=config.ROOT + "/workspace/log_file_dir",
                                    fifo_mode=True,
                                    delete_workdir=True,
                                    parallelize=True,
                                    logger=None
                                    ):
    """
    Run multiple QualityRunner in parallel.
    :param runner_class:
    :param assets:
    :param log_file_dir:
    :param fifo_mode:
    :param delete_workdir:
    :param parallelize:
    :return:
    """

    def run_quality_runner(args):
        runner_class, asset, log_file_dir, fifo_mode, delete_workdir = args
        runner = runner_class(
            [asset], None, log_file_dir, fifo_mode, delete_workdir)
        runner.run()
        return runner

    # pack key arguments to be used as inputs to map function
    list_args = []
    for asset in assets:
        list_args.append(
            [runner_class, asset, log_file_dir, fifo_mode, delete_workdir])

    # map arguments to func
    if parallelize:
        try:
            from pathos.pp_map import pp_map
            runners = pp_map(run_quality_runner, list_args)
        except ImportError:
            # fall back
            msg = "pathos.pp_map cannot be imported, fall back to sequential " \
                  "map(). On Ubuntu, install pathos by: \npip install pathos"
            if logger:
                logger.warn(msg)
            else:
                print 'Warn: {}'.format(msg)
            runners = map(run_quality_runner, list_args)
    else:
        runners = map(run_quality_runner, list_args)

    # aggregate results
    results = [runner.results[0] for runner in runners]

    return runners, results
