import os

from vmaf.core.feature_extractor import FeatureExtractor
from vmaf.tools.misc import make_absolute_path, run_process
from vmaf.tools.stats import ListStats

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import numpy as np

from vmaf.config import VmafConfig, VmafExternalConfig

class MatlabFeatureExtractor(FeatureExtractor):

    @classmethod
    def _assert_class(cls):
        # override Executor._assert_class
        super(MatlabFeatureExtractor, cls)._assert_class()
        VmafExternalConfig.get_and_assert_matlab()

class StrredFeatureExtractor(MatlabFeatureExtractor):

    TYPE = 'STRRED_feature'

    # VERSION = '1.0'
    # VERSION = '1.1' # fix matlab code where width and height are mistakenly swapped
    VERSION = '1.2' # fix minor frame and prev frame swap issue

    ATOM_FEATURES = ['srred', 'trred', ]

    DERIVED_ATOM_FEATURES = ['strred', ]

    MATLAB_WORKSPACE = VmafConfig.root_path('matlab', 'strred')

    @classmethod
    def _assert_an_asset(cls, asset):
        super(StrredFeatureExtractor, cls)._assert_an_asset(asset)
        assert asset.ref_yuv_type == 'yuv420p' and asset.dis_yuv_type == 'yuv420p', \
            'STRRED feature extractor only supports yuv420p for now.'

    def _generate_result(self, asset):
        # routine to call the command-line executable and generate quality
        # scores in the log file.

        ref_workfile_path = asset.ref_workfile_path
        dis_workfile_path = asset.dis_workfile_path
        log_file_path = self._get_log_file_path(asset)

        current_dir = os.getcwd() + '/'

        ref_workfile_path = make_absolute_path(ref_workfile_path, current_dir)
        dis_workfile_path = make_absolute_path(dis_workfile_path, current_dir)
        log_file_path = make_absolute_path(log_file_path, current_dir)

        quality_width, quality_height = asset.quality_width_height

        strred_cmd = '''{matlab} -nodisplay -nosplash -nodesktop -r "run_strred('{ref}', '{dis}', {h}, {w}); exit;" >> {log_file_path}'''.format(
            matlab=VmafExternalConfig.get_and_assert_matlab(),
            ref=ref_workfile_path,
            dis=dis_workfile_path,
            w=quality_width,
            h=quality_height,
            log_file_path=log_file_path,
        )
        if self.logger:
            self.logger.info(strred_cmd)

        os.chdir(self.MATLAB_WORKSPACE)
        run_process(strred_cmd, shell=True)
        os.chdir(current_dir)

    @classmethod
    def _post_process_result(cls, result):
        # override Executor._post_process_result

        def _strred(srred_trred):
            srred, trred = srred_trred
            try:
                return srred * trred
            except TypeError: # possible either srred or trred is None
                return None

        result = super(StrredFeatureExtractor, cls)._post_process_result(result)

        # calculate refvar and disvar from ref1st, ref2nd, dis1st, dis2nd
        srred_scores_key = cls.get_scores_key('srred')
        trred_scores_key = cls.get_scores_key('trred')
        strred_scores_key = cls.get_scores_key('strred')

        srred_scores = result.result_dict[srred_scores_key]
        trred_scores = result.result_dict[trred_scores_key]

        # compute strred scores
        # === Way One: consistent with VMAF framework, which is to multiply S and T scores per frame, then average
        # strred_scores = map(_strred, zip(srred_scores, trred_scores))
        # === Way Two: authentic way of calculating STRRED score: average first, then multiply ===
        assert len(srred_scores) == len(trred_scores)
        strred_scores = ListStats.nonemean(srred_scores) * ListStats.nonemean(trred_scores) * np.ones(len(srred_scores))

        result.result_dict[strred_scores_key] = strred_scores

        # validate
        for feature in cls.DERIVED_ATOM_FEATURES:
            assert cls.get_scores_key(feature) in result.result_dict

        return result


class SpEEDMatlabFeatureExtractor(MatlabFeatureExtractor):

    TYPE = 'SpEED_Matlab_feature'

    VERSION = '0.1'

    scale_list = [2, 3, 4]
    ATOM_FEATURES = []
    DERIVED_ATOM_FEATURES = []
    for scale_now in scale_list:
        ATOM_FEATURES.append('sspeed_' + str(scale_now))
        ATOM_FEATURES.append('tspeed_' + str(scale_now))
        DERIVED_ATOM_FEATURES.append('speed_' + str(scale_now))

    MATLAB_WORKSPACE = VmafConfig.root_path('matlab', 'SpEED')

    def _generate_result(self, asset):

        # routine to call the command-line executable and generate quality
        # scores in the log file.
        ref_workfile_path = asset.ref_workfile_path
        dis_workfile_path = asset.dis_workfile_path
        log_file_path = self._get_log_file_path(asset)
        current_dir = os.getcwd() + '/'
        ref_workfile_path = make_absolute_path(ref_workfile_path, current_dir)
        dis_workfile_path = make_absolute_path(dis_workfile_path, current_dir)
        log_file_path = make_absolute_path(log_file_path, current_dir)
        quality_width, quality_height = asset.quality_width_height
        speed_cmd = '''{matlab} -nodisplay -nosplash -nodesktop -r "run_speed('{ref}', '{dis}', {w}, {h}, {bands}, '{yuv_type}'); exit;" >> {log_file_path}'''.format(
            matlab=VmafExternalConfig.get_and_assert_matlab(),
            ref=ref_workfile_path,
            dis=dis_workfile_path,
            w=quality_width,
            h=quality_height,
            bands=self.scale_list,
            yuv_type=self._get_workfile_yuv_type(asset),
            log_file_path=log_file_path,
        )
        if self.logger:
            self.logger.info(speed_cmd)
        os.chdir(self.MATLAB_WORKSPACE)
        run_process(speed_cmd, shell=True)
        os.chdir(current_dir)

    @classmethod
    def _post_process_result(cls, result):

        # override Executor._post_process_result
        def _speed(sspeed_tspeed):
            sspeed, tspeed = sspeed_tspeed
            if sspeed is not None and tspeed is not None:
                return sspeed * tspeed
            elif sspeed is None:
                return tspeed
            elif tspeed is None:
                return sspeed
            else:
                return None

        result = super(SpEEDMatlabFeatureExtractor, cls)._post_process_result(result)
        for scale_now in cls.scale_list:
            sspeed_scale_now_scores_key = cls.get_scores_key('sspeed_' + str(scale_now))
            tspeed_scale_now_scores_key = cls.get_scores_key('tspeed_' + str(scale_now))
            speed_scale_now_scores_key = cls.get_scores_key('speed_' + str(scale_now))
            sspeed_scale_now_scores = result.result_dict[sspeed_scale_now_scores_key]
            tspeed_scale_now_scores = result.result_dict[tspeed_scale_now_scores_key]
            assert len(sspeed_scale_now_scores) == len(tspeed_scale_now_scores)
            # consistent with VMAF framework, which is to multiply S and T scores per frame, then average
            speed_scale_now_scores = map(_speed, zip(sspeed_scale_now_scores, tspeed_scale_now_scores))
            result.result_dict[speed_scale_now_scores_key] = speed_scale_now_scores

        # validate
        for feature in cls.DERIVED_ATOM_FEATURES:
            assert cls.get_scores_key(feature) in result.result_dict
        return result
