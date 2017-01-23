__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
from time import sleep
import multiprocessing

import numpy as np

from tools.misc import make_parent_dirs_if_nonexist, get_dir_without_last_slash, \
    match_any_files
from core.feature_extractor import FeatureExtractor
from tools.reader import YuvReader


class NorefFeatureExtractor(FeatureExtractor):
    """
    NorefFeatureExtractor is a subclass of FeatureExtractor. Any derived
    subclass of NorefFeatureExtractor cannot use the reference video of an
    input asset.
    """

    def _assert_paths(self, asset):
        # Override Executor._assert_paths to skip asserting on ref_path
        assert os.path.exists(asset.dis_path) or match_any_files(asset.dis_path), \
            "Distorted path {} does not exist.".format(asset.dis_path)

    def _wait_for_workfiles(self, asset):
        # Override Executor._wait_for_workfiles to skip ref_workfile_path
        # wait til workfile paths being generated
        for i in range(10):
            if os.path.exists(asset.dis_workfile_path):
                break
            sleep(0.1)
        else:
            raise RuntimeError("dis video workfile path {} is missing.".format(
                asset.dis_workfile_path))

    def _run_on_asset(self, asset):
        # Override Executor._run_on_asset to skip working on ref video

        if self.result_store:
            result = self.result_store.load(asset, self.executor_id)
        else:
            result = None

        # if result can be retrieved from result_store, skip log file
        # generation and reading result from log file, but directly return
        # return the retrieved result
        if result is not None:
            if self.logger:
                self.logger.info('{id} result exists. Skip {id} run.'.
                                 format(id=self.executor_id))
        else:

            if self.logger:
                self.logger.info('{id} result does\'t exist. Perform {id} '
                                 'calculation.'.format(id=self.executor_id))

            # at this stage, it is certain that asset.ref_path and
            # asset.dis_path will be used. must early determine that
            # they exists
            self._assert_paths(asset)

            # if no rescaling is involved, directly work on ref_path/dis_path,
            # instead of opening workfiles
            self._set_asset_use_path_as_workpath(asset)

            # remove workfiles if exist (do early here to avoid race condition
            # when ref path and dis path have some overlap)
            if asset.use_path_as_workpath:
                # do nothing
                pass
            else:
                self._close_dis_workfile(asset)

            log_file_path = self._get_log_file_path(asset)
            make_parent_dirs_if_nonexist(log_file_path)

            if asset.use_path_as_workpath:
                # do nothing
                pass
            else:
                if self.fifo_mode:
                    dis_p = multiprocessing.Process(target=self._open_dis_workfile,
                                                    args=(asset, True))
                    dis_p.start()
                    self._wait_for_workfiles(asset)
                else:
                    self._open_dis_workfile(asset, fifo_mode=False)

            self._prepare_log_file(asset)

            self._generate_result(asset)

            # clean up workfiles
            if self.delete_workdir:
                if asset.use_path_as_workpath:
                    # do nothing
                    pass
                else:
                    self._close_dis_workfile(asset)

            if self.logger:
                self.logger.info("Read {id} log file, get scores...".
                                 format(type=self.executor_id))

            # collect result from each asset's log file
            result = self._read_result(asset)

            # save result
            if self.result_store:
                self.result_store.save(result)

            # clean up workdir and log files in it
            if self.delete_workdir:

                # remove log file
                self._remove_log(asset)

                # remove dir
                log_file_path = self._get_log_file_path(asset)
                log_dir = get_dir_without_last_slash(log_file_path)
                try:
                    os.rmdir(log_dir)
                except OSError as e:
                    if e.errno == 39: # [Errno 39] Directory not empty
                        # VQM could generate an error file with non-critical
                        # information like: '3 File is longer than 15 seconds.
                        # Results will be calculated using first 15 seconds
                        # only.' In this case, want to keep this
                        # informational file and pass
                        pass

        result = self._post_process_result(result)

        return result

class MomentNorefFeatureExtractor(NorefFeatureExtractor):

    TYPE = "Moment_noref_feature"
    VERSION = "1.0" # python only

    ATOM_FEATURES = ['1st', '2nd', ] # order matters

    DERIVED_ATOM_FEATURES = ['var', ]

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate feature
        # scores in the log file.

        quality_w, quality_h = asset.quality_width_height
        with YuvReader(filepath=asset.dis_workfile_path, width=quality_w,
                       height=quality_h,
                       yuv_type=self._get_workfile_yuv_type(asset.yuv_type)) \
                as dis_yuv_reader:
            scores_mtx_list = []
            i = 0
            for dis_yuv in dis_yuv_reader:
                dis_y = dis_yuv[0]
                firstm = dis_y.mean()
                secondm = dis_y.var() + firstm**2
                scores_mtx_list.append(np.hstack(([firstm], [secondm])))
                i += 1
            scores_mtx = np.vstack(scores_mtx_list)

        # write scores_mtx to log file
        log_file_path = self._get_log_file_path(asset)
        with open(log_file_path, "wb") as log_file:
            np.save(log_file, scores_mtx)

    def _get_feature_scores(self, asset):
        # routine to read the feature scores from the log file, and return
        # the scores in a dictionary format.

        log_file_path = self._get_log_file_path(asset)
        with open(log_file_path, "rb") as log_file:
            scores_mtx = np.load(log_file)

        num_frm, num_features = scores_mtx.shape
        assert num_features == len(self.ATOM_FEATURES)

        feature_result = {}

        for idx, atom_feature in enumerate(self.ATOM_FEATURES):
            scores_key = self.get_scores_key(atom_feature)
            feature_result[scores_key] = list(scores_mtx[:, idx])

        return feature_result

    @classmethod
    def _post_process_result(cls, result):
        # override Executor._post_process_result

        result = super(MomentNorefFeatureExtractor, cls)._post_process_result(result)

        # calculate var from 1st, 2nd
        var_scores_key = cls.get_scores_key('var')
        first_scores_key = cls.get_scores_key('1st')
        second_scores_key = cls.get_scores_key('2nd')
        get_var = lambda (m1, m2): m2 - m1 * m1
        result.result_dict[var_scores_key] = \
            map(get_var, zip(result.result_dict[first_scores_key],
                             result.result_dict[second_scores_key]))

        # validate
        for feature in cls.DERIVED_ATOM_FEATURES:
            assert cls.get_scores_key(feature) in result.result_dict

        return result

