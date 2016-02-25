__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

from feature_extractor import FeatureExtractor
from result import BasicResult

class FeatureAssembler(object):
    """
    Assembles features for a input list of Assets on a input list of
    FeatureExtractors. For each asset, it outputs a BasicResult object.
    """

    def __init__(self, feature_dict, feature_option_dict, assets, logger,
                 log_file_dir, fifo_mode, delete_workdir, result_store, optional_dict=None):
        """
        :param feature_dict: in the format of:
        {FeatureExtractor_type:'all', ...}, or
        {FeatureExtractor_type:[atom_features,], ...}.
        For example, the below are valid feature dicts:
        {'VMAF_feature':'all', 'BRISQUE_feature':'all'},
        {'VMAF_feature':['vif', 'ansnr'], 'BRISQUE_feature':'all'}
        :param feature_option_dict: contains options to extract a particular
        feature, for example:
        {'VMAF_feature':{'force_extraction':True}, 'BRISQUE_feature':{}},
        :param assets:
        :param logger:
        :param log_file_dir:
        :param fifo_mode:
        :param delete_workdir:
        :param result_store:
        :param optional_dict:
        :return:
        """
        self.feature_dict = feature_dict
        self.feature_option_dict = feature_option_dict
        self.assets = assets
        self.logger = logger
        self.log_file_dir = log_file_dir
        self.fifo_mode = fifo_mode
        self.delete_workdir = delete_workdir
        self.result_store = result_store
        self.optional_dict = optional_dict

        self.type2results_dict = {}

    def run(self):
        """
        Do all the calculation here.
        :return:
        """

        # for each FeatureExtractor_type key in feature_dict, find the subclass
        # of FeatureExtractor, run, and put results in a dict
        for fextractor_type in self.feature_dict:
            fextractor = self._get_fextractor_instance(fextractor_type)
            fextractor.run()
            self.type2results_dict[fextractor_type] = fextractor.results

        # assemble an output dict with demanded atom features
        # atom_features_dict = self.fextractor_atom_features_dict
        result_dicts = [dict() for _ in self.assets]
        for fextractor_type in self.feature_dict:
            assert fextractor_type in self.type2results_dict
            for atom_feature in self._get_atom_features(fextractor_type):
                scores_key = self._get_scores_key(fextractor_type, atom_feature)
                for result_index, result in enumerate(self.type2results_dict[
                                                          fextractor_type]):
                    result_dicts[result_index][scores_key] = result[scores_key]

        self.results = map(
            lambda (asset, result_dict): BasicResult(asset, result_dict),
            zip(self.assets, result_dicts)
        )

    def remove_logs(self):
        """
        Remove all the intermediate log files if no need for inspection.
        :return:
        """
        for fextractor_type in self.feature_dict:
            fextractor = self._get_fextractor_instance(fextractor_type)
            fextractor.remove_logs()

    def remove_results(self):
        """
        Remove all relevant Results stored in ResultStore, which is specified
        at the constructor.
        :return:
        """
        for fextractor_type in self.feature_dict:
            fextractor = self._get_fextractor_instance(fextractor_type)
            fextractor.remove_results()

    def _get_scores_key(self, fextractor_type, atom_feature):
        fextractor_subclass = FeatureExtractor.find_subclass(fextractor_type)
        scores_key = fextractor_subclass.get_scores_key(atom_feature)
        return scores_key

    def _get_atom_features(self, fextractor_type):
        if self.feature_dict[fextractor_type] == 'all':
            fextractor_class = FeatureExtractor.find_subclass(fextractor_type)
            atom_features = fextractor_class.ATOM_FEATURES
        else:
            atom_features = self.feature_dict[fextractor_type]
        return atom_features

    def _get_fextractor_instance(self, fextractor_type):
        fextractor_class = FeatureExtractor.find_subclass(fextractor_type)
        fextractor = fextractor_class(assets=self.assets,
                                      logger=self.logger,
                                      log_file_dir=self.log_file_dir,
                                      fifo_mode=self.fifo_mode,
                                      delete_workdir=self.delete_workdir,
                                      result_store=self.result_store,
                                      optional_dict=self.optional_dict
                                      )
        return fextractor

