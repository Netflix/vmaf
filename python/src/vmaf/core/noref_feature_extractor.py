from vmaf.core.executor import NorefExecutorMixin

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import numpy as np

from vmaf.core.feature_extractor import FeatureExtractor
from vmaf.tools.reader import YuvReader


class MomentNorefFeatureExtractor(NorefExecutorMixin, FeatureExtractor):

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
                       yuv_type=self._get_workfile_yuv_type(asset)) \
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
