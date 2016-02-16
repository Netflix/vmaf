__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import config
from quality_runner import QualityRunner
from functools import partial
from vmaf_feature_extractor import VmafFeatureExtractor

class VmafQualityRunner(QualityRunner):

    TYPE = 'VMAF'
    VERSION = '0.1'

    SVM_MODEL_FILE = config.ROOT + "/resource/model/model_V8a.model"

    FEAT_RESCALE = {'vif': (0.0, 1.0), 'adm': (0.4, 1.0),
                    'ansnr': (10.0, 50.0), 'motion': (0.0, 20.0)}

    sys.path.append(config.ROOT + "/libsvm/python")
    import svmutil

    asset2quality_map = {}

    # override Executor._run_on_asset
    def _run_on_asset(self, asset):
        """
        Override Executor._run_on_asset to bypass calling
        _run_and_generate_log_file() itself. Instead, initiate a
        VmafFeatureExtractor object and run, which will do the work.
        :param asset:
        :return:
        """
        vmaf_fextractor = VmafFeatureExtractor([asset],
                                               self.logger,
                                               self.log_file_dir,
                                               self.fifo_mode,
                                               self.delete_workdir,
                                               self.result_store)
        vmaf_fextractor.run()
        feature_result = vmaf_fextractor.results[0]

        # SVR predict
        model = self.svmutil.svm_load_model(self.SVM_MODEL_FILE)

        scaled_vif_scores = self._rescale(
            feature_result[vmaf_fextractor.TYPE + '_vif_scores'], self.FEAT_RESCALE['vif'])
        scaled_adm_scores = self._rescale(
            feature_result[vmaf_fextractor.TYPE + '_adm_scores'], self.FEAT_RESCALE['adm'])
        scaled_ansnr_scores = self._rescale(
            feature_result[vmaf_fextractor.TYPE + '_ansnr_scores'], self.FEAT_RESCALE['ansnr'])
        scaled_motion_scores = self._rescale(
            feature_result[vmaf_fextractor.TYPE + '_motion_scores'], self.FEAT_RESCALE['motion'])

        scores = []
        for vif, adm, ansnr, motion in zip(scaled_vif_scores,
                                           scaled_adm_scores,
                                           scaled_ansnr_scores,
                                           scaled_motion_scores):
            xs = [[vif, adm, ansnr, motion]]
            score = self.svmutil.svm_predict([0], xs, model)[0][0]
            score = self._post_correction(motion, score)
            scores.append(score)

        quality_result = {}

        # add all feature result
        quality_result.update(feature_result.result_dict)

        # add quality score
        quality_result[self.TYPE + '_scores'] = scores

        # save to asset2quality map
        self.asset2quality_map[repr(asset)] = quality_result

        result = self._read_result(asset)

        return result

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
        """
        Since result already stored in asset2quality map, just retrieve it
        :param asset:
        :return:
        """
        repr_asset = repr(asset)
        assert repr_asset in self.asset2quality_map
        return self.asset2quality_map[repr_asset]

    def _remove_log(self, asset):
        """
        Remove VmafFeatureExtractor's log instead
        :param asset:
        :return:
        """
        vmaf_fextractor = VmafFeatureExtractor([asset],
                                               self.logger,
                                               self.log_file_dir,
                                               self.fifo_mode,
                                               self.delete_workdir,
                                               self.result_store)
        vmaf_fextractor._remove_log(asset)

    def _remove_result(self, asset):
        """
        Remove VmafFeatureExtractor's result instead
        :param asset:
        :return:
        """
        vmaf_fextractor = VmafFeatureExtractor([asset],
                                               self.logger,
                                               self.log_file_dir,
                                               self.fifo_mode,
                                               self.delete_workdir,
                                               self.result_store)
        vmaf_fextractor._remove_result(asset)