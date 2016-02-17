__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import sys
import config
from quality_runner import QualityRunner
import numpy as np
from feature_assembler import FeatureAssembler

class VmafQualityRunner(QualityRunner):

    TYPE = 'VMAF'
    VERSION = '0.1'

    FEATURE_ASSEMBLER_DICT = {'VMAF_feature': 'all'}

    FEATURE_RESCALE = {'VMAF_feature_vif_scores': (0.0, 1.0),
                       'VMAF_feature_adm_scores': (0.4, 1.0),
                       'VMAF_feature_ansnr_scores': (10.0, 50.0),
                       'VMAF_feature_motion_scores': (0.0, 20.0)}

    SVM_MODEL_FILE = config.ROOT + "/resource/model/model_V8a.model"
    SVM_MODEL_ORDERED_SCORES_KEYS = ['VMAF_feature_vif_scores',
                                     'VMAF_feature_adm_scores',
                                     'VMAF_feature_ansnr_scores',
                                     'VMAF_feature_motion_scores']

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
        vmaf_fassembler = self._get_vmaf_feature_assembler_instance(asset)
        vmaf_fassembler.run()
        feature_result = vmaf_fassembler.results[0]

        assert feature_result.asset == asset

        # SVR predict
        model = self.svmutil.svm_load_model(self.SVM_MODEL_FILE)

        ordered_scaled_scores_list = []
        for scores_key in self.SVM_MODEL_ORDERED_SCORES_KEYS:
            scaled_scores = self._rescale(feature_result[scores_key],
                                          self.FEATURE_RESCALE[scores_key])
            ordered_scaled_scores_list.append(scaled_scores)

        scores = []
        for score_vector in zip(*ordered_scaled_scores_list):
            vif, adm, ansnr, motion = score_vector
            xs = [[vif, adm, ansnr, motion]]
            score = self.svmutil.svm_predict([0], xs, model)[0][0]
            score = self._post_correction(motion, score)
            scores.append(score)

        quality_result = {}

        # add all feature result
        quality_result.update(feature_result.result_dict)

        # add quality score
        quality_result[self._get_scores_key()] = scores

        # save to asset2quality map
        self.asset2quality_map[repr(asset)] = quality_result

        result = self._read_result(asset)

        return result

    def _get_vmaf_feature_assembler_instance(self, asset):
        vmaf_fassembler = FeatureAssembler(
            feature_dict=self.FEATURE_ASSEMBLER_DICT,
            assets=[asset],
            logger=self.logger,
            log_file_dir=self.log_file_dir,
            fifo_mode=self.fifo_mode,
            delete_workdir=self.delete_workdir,
            result_store=self.result_store
        )
        return vmaf_fassembler

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
        lower_bound, upper_bound = lower_upper_bound
        vals = np.double(vals)
        vals = np.clip(vals, lower_bound, upper_bound)
        vals = (vals - lower_bound)/(upper_bound - lower_bound)
        return vals

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
        vmaf_fassembler = self._get_vmaf_feature_assembler_instance(asset)
        vmaf_fassembler.remove_logs()

    def _remove_result(self, asset):
        """
        Remove VmafFeatureExtractor's result instead
        :param asset:
        :return:
        """
        vmaf_fassembler = self._get_vmaf_feature_assembler_instance(asset)
        vmaf_fassembler.remove_results()
