import os

from vmaf.config import VmafExternalConfig, VmafConfig
from vmaf.tools.misc import run_process
from vmaf.core.feature_assembler import FeatureAssembler
from vmaf.core.matlab_feature_extractor import StrredFeatureExtractor, StrredOptFeatureExtractor, SpEEDMatlabFeatureExtractor, STMADFeatureExtractor
from vmaf.core.quality_runner import QualityRunner
from vmaf.core.result import Result

__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"


class StrredQualityRunner(QualityRunner):

    TYPE = 'STRRED'

    # VERSION = '1.0'
    VERSION = 'F' + StrredFeatureExtractor.VERSION + '-1.1'

    def _get_quality_scores(self, asset):
        raise NotImplementedError

    def _generate_result(self, asset):
        raise NotImplementedError

    def _get_feature_assembler_instance(self, asset):

        feature_dict = {StrredFeatureExtractor.TYPE: StrredFeatureExtractor.ATOM_FEATURES + getattr(StrredFeatureExtractor, 'DERIVED_ATOM_FEATURES', [])}

        feature_assembler = FeatureAssembler(
            feature_dict=feature_dict,
            feature_option_dict=None,
            assets=[asset],
            logger=self.logger,
            fifo_mode=self.fifo_mode,
            delete_workdir=self.delete_workdir,
            result_store=self.result_store,
            optional_dict=None,
            optional_dict2=None,
            parallelize=False, # parallelization already in a higher level
        )
        return feature_assembler

    def _run_on_asset(self, asset):
        # Override Executor._run_on_asset(self, asset)
        vmaf_fassembler = self._get_feature_assembler_instance(asset)
        vmaf_fassembler.run()
        feature_result = vmaf_fassembler.results[0]
        result_dict = {}
        result_dict.update(feature_result.result_dict.copy()) # add feature result
        result_dict[self.get_scores_key()] = feature_result.result_dict[
            StrredFeatureExtractor.get_scores_key('strred')] # add strred score
        del result_dict[StrredFeatureExtractor.get_scores_key('strred')] # delete redundant
        return Result(asset, self.executor_id, result_dict)

    def _remove_result(self, asset):
        # Override Executor._remove_result(self, asset) by redirecting it to the
        # FeatureAssembler.

        vmaf_fassembler = self._get_feature_assembler_instance(asset)
        vmaf_fassembler.remove_results()


class StrredOptQualityRunner(QualityRunner):

    TYPE = 'STRREDOpt'

    VERSION = 'F' + StrredOptFeatureExtractor.VERSION + '-1.1'

    def _get_quality_scores(self, asset):
        raise NotImplementedError

    def _generate_result(self, asset):
        raise NotImplementedError

    def _get_feature_assembler_instance(self, asset):

        feature_dict = {StrredOptFeatureExtractor.TYPE: StrredOptFeatureExtractor.ATOM_FEATURES + getattr(StrredOptFeatureExtractor, 'DERIVED_ATOM_FEATURES', [])}

        feature_assembler = FeatureAssembler(
            feature_dict=feature_dict,
            feature_option_dict=None,
            assets=[asset],
            logger=self.logger,
            fifo_mode=self.fifo_mode,
            delete_workdir=self.delete_workdir,
            result_store=self.result_store,
            optional_dict=None,
            optional_dict2=None,
            parallelize=False, # parallelization already in a higher level
        )
        return feature_assembler

    def _run_on_asset(self, asset):
        # Override Executor._run_on_asset(self, asset)
        vmaf_fassembler = self._get_feature_assembler_instance(asset)
        vmaf_fassembler.run()
        feature_result = vmaf_fassembler.results[0]
        result_dict = {}
        result_dict.update(feature_result.result_dict.copy()) # add feature result
        result_dict[self.get_scores_key()] = feature_result.result_dict[
            StrredOptFeatureExtractor.get_scores_key('strred')] # add strred score
        del result_dict[StrredOptFeatureExtractor.get_scores_key('strred')] # delete redundant
        return Result(asset, self.executor_id, result_dict)

    def _remove_result(self, asset):
        # Override Executor._remove_result(self, asset) by redirecting it to the
        # FeatureAssembler.

        vmaf_fassembler = self._get_feature_assembler_instance(asset)
        vmaf_fassembler.remove_results()


class SpEEDMatlabQualityRunner(QualityRunner):

        TYPE = 'SpEED_Matlab'

        # VERSION = '1.0'
        VERSION = 'F' + SpEEDMatlabFeatureExtractor.VERSION + '-1.1'

        def _get_quality_scores(self, asset):
            raise NotImplementedError

        def _generate_result(self, asset):
            raise NotImplementedError

        def _get_feature_assembler_instance(self, asset):

            feature_dict = {SpEEDMatlabFeatureExtractor.TYPE: SpEEDMatlabFeatureExtractor.ATOM_FEATURES + getattr(
                SpEEDMatlabFeatureExtractor, 'DERIVED_ATOM_FEATURES', [])}

            feature_assembler = FeatureAssembler(
                feature_dict=feature_dict,
                feature_option_dict=None,
                assets=[asset],
                logger=self.logger,
                fifo_mode=self.fifo_mode,
                delete_workdir=self.delete_workdir,
                result_store=self.result_store,
                optional_dict=None,
                optional_dict2=None,
                parallelize=False,  # parallelization already in a higher level
            )
            return feature_assembler

        def _run_on_asset(self, asset):
            # Override Executor._run_on_asset(self, asset)
            speed_fassembler = self._get_feature_assembler_instance(asset)
            speed_fassembler.run()
            feature_result = speed_fassembler.results[0]
            result_dict = {}
            result_dict.update(feature_result.result_dict.copy())  # add feature result

            result_dict[self.get_scores_key()] = feature_result.result_dict[
                SpEEDMatlabFeatureExtractor.get_scores_key('speed_4')]  # add SpEED score at scale 4

            return Result(asset, self.executor_id, result_dict)

        def _remove_result(self, asset):
            # Override Executor._remove_result(self, asset) by redirecting it to the
            # FeatureAssembler.

            speed_fassembler = self._get_feature_assembler_instance(asset)
            speed_fassembler.remove_results()


class STMADQualityRunner(QualityRunner):

    TYPE = 'STMAD'

    VERSION = 'F' + STMADFeatureExtractor.VERSION + '-1.1'

    def _get_quality_scores(self, asset):
        raise NotImplementedError

    def _generate_result(self, asset):
        raise NotImplementedError

    def _get_feature_assembler_instance(self, asset):
        feature_dict = {STMADFeatureExtractor.TYPE: STMADFeatureExtractor.ATOM_FEATURES + getattr(
            STMADFeatureExtractor, 'DERIVED_ATOM_FEATURES', [])}

        feature_assembler = FeatureAssembler(
            feature_dict=feature_dict,
            feature_option_dict=None,
            assets=[asset],
            logger=self.logger,
            fifo_mode=self.fifo_mode,
            delete_workdir=self.delete_workdir,
            result_store=self.result_store,
            optional_dict=None,
            optional_dict2=None,
            parallelize=False,  # parallelization already in a higher level
        )
        return feature_assembler

    def _run_on_asset(self, asset):
        # Override Executor._run_on_asset(self, asset)
        vmaf_fassembler = self._get_feature_assembler_instance(asset)
        vmaf_fassembler.run()
        feature_result = vmaf_fassembler.results[0]
        result_dict = {}
        result_dict.update(feature_result.result_dict.copy())  # add feature result
        result_dict[self.get_scores_key()] = feature_result.result_dict[
            STMADFeatureExtractor.get_scores_key('stmad')]  # add strred score
        del result_dict[STMADFeatureExtractor.get_scores_key('stmad')]  # delete redundant
        return Result(asset, self.executor_id, result_dict)

    def _remove_result(self, asset):
        # Override Executor._remove_result(self, asset) by redirecting it to the
        # FeatureAssembler.

        vmaf_fassembler = self._get_feature_assembler_instance(asset)
        vmaf_fassembler.remove_results()
