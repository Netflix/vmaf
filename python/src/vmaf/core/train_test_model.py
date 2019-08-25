from abc import ABCMeta, abstractmethod
import os
import pickle
from numbers import Number

from sklearn.metrics import f1_score
import numpy as np

from vmaf import to_list
from vmaf.tools.decorator import deprecated
from vmaf.tools.misc import indices
from vmaf.core.mixin import TypeVersionEnabled
from vmaf.core.perf_metric import RmsePerfMetric, SrccPerfMetric, PccPerfMetric, \
    KendallPerfMetric, AucPerfMetric, ResolvingPowerPerfMetric

__copyright__ = "Copyright 2016-2018, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class RegressorMixin(object):

    @classmethod
    def get_stats(cls, ys_label, ys_label_pred, **kwargs):

        # cannot have None
        assert all(x is not None for x in ys_label)
        assert all(x is not None for x in ys_label_pred)

        # RMSE
        rmse = RmsePerfMetric(ys_label, ys_label_pred) \
            .evaluate(enable_mapping=True)['score']

        # spearman
        srcc = SrccPerfMetric(ys_label, ys_label_pred) \
            .evaluate(enable_mapping=True)['score']

        # pearson
        pcc = PccPerfMetric(ys_label, ys_label_pred) \
            .evaluate(enable_mapping=True)['score']

        # kendall
        kendall = KendallPerfMetric(ys_label, ys_label_pred) \
            .evaluate(enable_mapping=True)['score']

        stats = {'RMSE': rmse,
                 'SRCC': srcc,
                 'PCC': pcc,
                 'KENDALL': kendall,
                 'ys_label': list(ys_label),
                 'ys_label_pred': list(ys_label_pred)}

        # create perf metric distributions, if multiple predictions are passed in as kwargs
        # spearman distribution for now
        if 'ys_label_pred_all_models' in kwargs:

            ys_label_pred_all_models = kwargs['ys_label_pred_all_models']

            srcc_all_models = []
            pcc_all_models = []
            rmse_all_models = []

            for ys_label_pred_some_model in ys_label_pred_all_models:
                srcc_some_model = SrccPerfMetric(ys_label, ys_label_pred_some_model) \
                    .evaluate(enable_mapping=True)['score']
                pcc_some_model = PccPerfMetric(ys_label, ys_label_pred_some_model) \
                    .evaluate(enable_mapping=True)['score']
                rmse_some_model = RmsePerfMetric(ys_label, ys_label_pred_some_model) \
                    .evaluate(enable_mapping=True)['score']
                srcc_all_models.append(srcc_some_model)
                pcc_all_models.append(pcc_some_model)
                rmse_all_models.append(rmse_some_model)

            stats['SRCC_across_model_distribution'] = srcc_all_models
            stats['PCC_across_model_distribution'] = pcc_all_models
            stats['RMSE_across_model_distribution'] = rmse_all_models

        ys_label_raw = kwargs['ys_label_raw'] if 'ys_label_raw' in kwargs else None

        if ys_label_raw is not None:
            try:
                # AUC
                result = AucPerfMetric(ys_label_raw, ys_label_pred).evaluate()
                stats['AUC_DS'] = result['AUC_DS']
                stats['AUC_BW'] = result['AUC_BW']
            except TypeError: # AUC would not work with dictionary-style dataset
                stats['AUC_DS'] = float('nan')
                stats['AUC_BW'] = float('nan')

            try:
                # ResPow
                respow = ResolvingPowerPerfMetric(ys_label_raw, ys_label_pred) \
                    .evaluate(enable_mapping=False)['score']
                stats['ResPow'] = respow
            except TypeError: # ResPow would not work with dictionary-style dataset
                stats['ResPow'] = float('nan')

            try:
                # ResPow
                respow_norm = ResolvingPowerPerfMetric(ys_label_raw, ys_label_pred) \
                    .evaluate(enable_mapping=True)['score']
                stats['ResPowNormalized'] = respow_norm
            except TypeError: # ResPow would not work with dictionary-style dataset
                stats['ResPowNormalized'] = float('nan')

        if 'ys_label_stddev' in kwargs and 'ys_label_stddev' and kwargs['ys_label_stddev'] is not None:
            stats['ys_label_stddev'] = kwargs['ys_label_stddev']

        return stats

    @staticmethod
    def format_stats_for_plot(stats):
        if stats is None:
            return '(Invalid Stats)'
        else:
            if 'AUC_DS' in stats and 'AUC_BW' in stats and 'ResPow' in stats and 'ResPowNormalized' in stats:
                return '(SRCC: {srcc:.3f}, PCC: {pcc:.3f}, RMSE: {rmse:.3f},\n AUC: {auc_ds:.3f}/{auc_bw:.3f}, ' \
                       'ResPow: {respow:.3f}/{respownorm:.3f})'. \
                    format(srcc=stats['SRCC'], pcc=stats['PCC'], rmse=stats['RMSE'],
                           auc_ds=stats['AUC_DS'], auc_bw=stats['AUC_BW'],
                           respow=stats['ResPow'], respownorm=stats['ResPowNormalized'])
            else:
                return '(SRCC: {srcc:.3f}, PCC: {pcc:.3f}, RMSE: {rmse:.3f})'. \
                    format(srcc=stats['SRCC'], pcc=stats['PCC'], rmse=stats['RMSE'])

    @staticmethod
    def format_stats_for_print(stats):
        if stats is None:
            return '(Invalid Stats)'
        else:
            if 'AUC_DS' in stats and 'AUC_BW' in stats and 'ResPow' in stats and 'ResPowNormalized' in stats:
                return '(SRCC: {srcc:.3f}, PCC: {pcc:.3f}, RMSE: {rmse:.3f}, AUC: {auc_ds:.3f}/{auc_bw:.3f}, ' \
                       'ResPow: {respow:.3f}/{respownorm:.3f})'. \
                    format(srcc=stats['SRCC'], pcc=stats['PCC'], rmse=stats['RMSE'],
                           auc_ds=stats['AUC_DS'], auc_bw=stats['AUC_BW'],
                           respow=stats['ResPow'], respownorm=stats['ResPowNormalized'])
            else:
                return '(SRCC: {srcc:.3f}, PCC: {pcc:.3f}, RMSE: {rmse:.3f})'. \
                    format(srcc=stats['SRCC'], pcc=stats['PCC'], rmse=stats['RMSE'])

    @staticmethod
    def format_across_model_stats_for_print(stats):
        if stats is None:
            return '(Invalid Stats)'
        else:
            return '(SRCC: {srcc:.3f}+/-{srcc_ci:.3f}, PCC: {pcc:.3f}+/-{pcc_ci:.3f}, RMSE: {rmse:.3f})+/-{rmse_ci:.3f}'. \
                format(srcc=stats['SRCC'], srcc_ci=stats['SRCC_across_model_ci'],
                       pcc=stats['PCC'], pcc_ci=stats['PCC_across_model_ci'],
                       rmse=stats['RMSE'], rmse_ci=stats['RMSE_across_model_ci'], )

    @staticmethod
    def extract_across_model_stats(stats):
        if 'SRCC_across_model_distribution' in stats \
                and 'PCC_across_model_distribution' in stats\
                and 'RMSE_across_model_distribution' in stats:
            srcc_across_model_ci = 1.96 * np.std(stats['SRCC_across_model_distribution'])
            pcc_across_model_ci = 1.96 * np.std(stats['PCC_across_model_distribution'])
            rmse_across_model_ci = 1.96 * np.std(stats['RMSE_across_model_distribution'])
            stats['SRCC_across_model_ci'] = srcc_across_model_ci
            stats['PCC_across_model_ci'] = pcc_across_model_ci
            stats['RMSE_across_model_ci'] = rmse_across_model_ci
        return stats

    @staticmethod
    @deprecated
    def format_stats2(stats):
        if stats is None:
            return 'Invalid Stats'
        else:
            return 'RMSE: {rmse:.3f}\nPCC: {pcc:.3f}\nSRCC: {srcc:.3f}'.format(
                srcc=stats['SRCC'], pcc=stats['PCC'], rmse=stats['RMSE'])

    @classmethod
    def aggregate_stats_list(cls, stats_list):
        aggregate_ys_label = []
        aggregate_ys_label_pred = []
        for stats in stats_list:
            aggregate_ys_label += stats['ys_label']
            aggregate_ys_label_pred += stats['ys_label_pred']
        return cls.get_stats(aggregate_ys_label, aggregate_ys_label_pred)

    @classmethod
    def plot_scatter(cls, ax, stats, **kwargs):

        assert len(stats['ys_label']) == len(stats['ys_label_pred'])

        content_ids = kwargs['content_ids'] if 'content_ids' in kwargs else None
        point_labels = kwargs['point_labels'] if 'point_labels' in kwargs else None

        if content_ids is None:
            ax.scatter(stats['ys_label'], stats['ys_label_pred'])
        else:
            assert len(stats['ys_label']) == len(content_ids)

            unique_content_ids = list(set(content_ids))
            from vmaf import plt
            cmap = plt.get_cmap()
            colors = [cmap(i) for i in np.linspace(0, 1, len(unique_content_ids))]
            for idx, curr_content_id in enumerate(unique_content_ids):
                curr_idxs = indices(content_ids, lambda cid: cid == curr_content_id)
                curr_ys_label = np.array(stats['ys_label'])[curr_idxs]
                curr_ys_label_pred = np.array(stats['ys_label_pred'])[curr_idxs]
                try:
                    curr_ys_label_stddev = np.array(stats['ys_label_stddev'])[curr_idxs]
                    ax.errorbar(curr_ys_label, curr_ys_label_pred,
                                xerr=1.96 * curr_ys_label_stddev,
                                marker='o', linestyle='', label=curr_content_id, color=colors[idx % len(colors)])
                except:
                    ax.errorbar(curr_ys_label, curr_ys_label_pred,
                                marker='o', linestyle='', label=curr_content_id, color=colors[idx % len(colors)])

        if point_labels:
            assert len(point_labels) == len(stats['ys_label'])
            for i, point_label in enumerate(point_labels):
                ax.annotate(point_label, (stats['ys_label'][i], stats['ys_label_pred'][i]))

    @staticmethod
    def get_objective_score(result, type='SRCC'):
        """
        Objective score is something to MAXIMIZE. e.g. SRCC, or -RMSE.
        :param result:
        :param type:
        :return:
        """
        if type == 'SRCC':
            return result['SRCC']
        elif type == 'PCC':
            return result['PCC']
        elif type == 'KENDALL':
            return result['KENDALL']
        elif type == 'RMSE':
            return -result['RMSE']
        else:
            assert False, 'Unknow type: {} for get_objective_score().'.format(type)

class ClassifierMixin(object):

    @classmethod
    def get_stats(cls, ys_label, ys_label_pred, **kwargs):

        # cannot have None
        assert all(x is not None for x in ys_label)
        assert all(x is not None for x in ys_label_pred)

        # RMSE
        rmse = np.sqrt(np.mean(
            np.power(np.array(ys_label) - np.array(ys_label_pred), 2.0)))
        # f1
        f1 = f1_score(ys_label_pred, ys_label)
        # error rate
        errorrate = np.mean(np.array(ys_label) != np.array(ys_label_pred))
        stats = {'RMSE': rmse,
                 'f1': f1,
                 'errorrate': errorrate,
                 'ys_label': list(ys_label),
                 'ys_label_pred': list(ys_label_pred)}
        return stats

    @staticmethod
    def format_stats(stats):
        if stats is None:
            return '(Invalid Stats)'
        else:
            return '(F1: {f1:.3f}, Error: {err:.3f}, RMSE: {rmse:.3f})'.format(
                f1=stats['f1'], err=stats['errorrate'], rmse=stats['RMSE'])

    @staticmethod
    def format_stats2(stats):
        if stats is None:
            return 'Invalid Stats'
        else:
            return 'RMSE: {rmse:.3f}\nF1: {f1:.3f}\nError: {err:.3f}'.format(
                f1=stats['f1'], err=stats['errorrate'], rmse=stats['RMSE'])

    @classmethod
    def aggregate_stats_list(cls, stats_list):
        aggregate_ys_label = []
        aggregate_ys_label_pred = []
        for stats in stats_list:
            aggregate_ys_label += stats['ys_label']
            aggregate_ys_label_pred += stats['ys_label_pred']
        return cls.get_stats(aggregate_ys_label, aggregate_ys_label_pred)

    @staticmethod
    def get_objective_score(result, type='RMSE'):
        """
        Objective score is something to MAXIMIZE. e.g. f1, or -errorrate, or -RMSE.
        :param result:
        :param type:
        :return:
        """
        if type == 'f1':
            return result['f1']
        elif type == 'errorrate':
            return -result['errorrate']
        elif type == 'RMSE':
            return -result['RMSE']
        else:
            assert False, 'Unknow type: {} for get_objective_score().'.format(type)


class TrainTestModel(TypeVersionEnabled):

    __metaclass__ = ABCMeta

    @classmethod
    @abstractmethod
    def _train(cls, param_dict, xys_2d, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _predict(cls, model, xs_2d):
        raise NotImplementedError

    def __init__(self, param_dict, logger=None, optional_dict2=None):
        '''
        Put in optional_dict2 optionals that would not impact result, e.g.
        path to checkpoint file directories, or h5py file
        '''
        TypeVersionEnabled.__init__(self)
        self.param_dict = param_dict
        self.logger = logger
        self.optional_dict2 = optional_dict2

        self.model_dict = {}

        self._assert_args()

    def _assert_args(self):
        pass

    @property
    def model_id(self):
        return TypeVersionEnabled.get_type_version_string(self)

    def _assert_trained(self):
        assert 'model_type' in self.model_dict # need this to recover class
        assert 'feature_names' in self.model_dict
        assert 'norm_type' in self.model_dict
        assert 'model' in self.model_dict

        norm_type = self.model_dict['norm_type']
        assert norm_type == 'none' or norm_type == 'linear_rescale'

        if norm_type == 'linear_rescale':
            assert 'slopes' in self.model_dict
            assert 'intercepts' in self.model_dict

    def append_info(self, key, value):
        """
        Useful for adding extra info to model before saving. For example,
        save feature_dict to model so that when the model is loaded by a
        QualityRunner, it knows when features to extract.
        """
        self.model_dict[key] = value

    def get_appended_info(self, key):
        """
        Retrieve info added via the append_info method.
        """
        return self.model_dict[key] if key in self.model_dict else None

    @property
    def feature_names(self):
        self._assert_trained()
        return self.model_dict['feature_names']

    @feature_names.setter
    def feature_names(self, value):
        self.model_dict['feature_names'] = value

    @property
    def model_type(self):
        return self.model_dict['model_type']

    @model_type.setter
    def model_type(self, value):
        self.model_dict['model_type'] = value

    @property
    def norm_type(self):
        return self.model_dict['norm_type']

    @norm_type.setter
    def norm_type(self, value):
        self.model_dict['norm_type'] = value

    @property
    def mus(self):
        return np.array(self.model_dict['mus'])

    @mus.setter
    def mus(self, value):
        # forcing float, to be used by PicklingTools and read in C++
        self.model_dict['mus'] = to_list(map(lambda x: float(x), list(value)))

    @property
    def sds(self):
        return np.array(self.model_dict['sds'])

    @sds.setter
    def sds(self, value):
        # forcing float, to be used by PicklingTools and read in C++
        self.model_dict['sds'] = to_list(map(lambda x: float(x), list(value)))

    @property
    def slopes(self):
        return np.array(self.model_dict['slopes'])

    @slopes.setter
    def slopes(self, value):
        # forcing float, to be used by PicklingTools and read in C++
        self.model_dict['slopes'] = to_list(map(lambda x: float(x), list(value)))

    @property
    def intercepts(self):
        return np.array(self.model_dict['intercepts'])

    @intercepts.setter
    def intercepts(self, value):
        # forcing float, to be used by PicklingTools and read in C++
        self.model_dict['intercepts'] = to_list(map(lambda x: float(x), list(value)))

    @property
    def model(self):
        return self.model_dict['model']

    @model.setter
    def model(self, value):
        self.model_dict['model'] = value

    def to_file(self, filename):
        self._assert_trained()
        param_dict = self.param_dict
        model_dict = self.model_dict
        self._to_file(filename, param_dict, model_dict)

    @staticmethod
    def _to_file(filename, param_dict, model_dict):
        info_to_save = {'param_dict': param_dict,
                        'model_dict': model_dict}
        with open(filename, 'wb') as file:
            pickle.dump(info_to_save, file)

    @classmethod
    def from_file(cls, filename, logger=None, optional_dict2=None):
        assert os.path.exists(filename), 'File name {} does not exist.'.format(filename)
        with open(filename, 'rb') as file:
            info_loaded = pickle.load(file)
        model_type = info_loaded['model_dict']['model_type']
        model_class = TrainTestModel.find_subclass(model_type)
        if model_class == cls:
            train_test_model = model_class._from_info_loaded(info_loaded, filename,
                                                             logger, optional_dict2)
        else:
            # the newly found model_class can be a different class (e.g. a subclass of cls). In this
            # case, call from_file() of that model_class.
            train_test_model = model_class.from_file(filename, logger, optional_dict2)

        return train_test_model

    @classmethod
    def _from_info_loaded(cls, info_loaded, filename, logger, optional_dict2):
        train_test_model = cls(
            param_dict={}, logger=logger, optional_dict2=optional_dict2)
        train_test_model.param_dict = info_loaded['param_dict']
        train_test_model.model_dict = info_loaded['model_dict']
        return train_test_model

    def _preproc_train(self, xys):
        self.model_type = self.TYPE
        assert 'label' in xys
        assert 'content_id' in xys
        feature_names = self.get_ordered_feature_names(xys)
        self.feature_names = feature_names
        # note that feature_names is property (write). below cannot yet use
        # self.feature_names since additional things (_assert_trained()) is
        # not ready yet
        xys_2d = self._to_tabular_xys(feature_names, xys)
        # calculate normalization parameters,
        self._calculate_normalization_params(xys_2d)
        # normalize
        xys_2d = self._normalize_xys(xys_2d)
        return xys_2d

    def train(self, xys, **kwargs):
        xys_2d = self._preproc_train(xys)
        model = self._train(self.param_dict, xys_2d, **kwargs)
        self.model = model

    @staticmethod
    def get_ordered_feature_names(xys_or_xs):
        # this makes sure the order of features are normalized, and each
        # dimension of xys_2d (or xs_2d) is consistent with feature_names
        feature_names = sorted(xys_or_xs.keys())
        if 'label' in feature_names:
            feature_names.remove('label')
        if 'content_id' in feature_names:
            feature_names.remove('content_id')
        return feature_names

    def _calculate_normalization_params(self, xys_2d):

        norm_type = self.param_dict['norm_type'] \
            if 'norm_type' in self.param_dict else 'none'

        if norm_type == 'normalize':
            mus = np.mean(xys_2d, axis=0)
            sds = np.std(xys_2d, axis=0)
            self.slopes = 1.0 / sds
            self.intercepts = - mus / sds
            self.norm_type = 'linear_rescale'
        elif norm_type == 'clip_0to1':
            self._calculate_normalization_params_clip_0to1(xys_2d)
        elif norm_type == 'custom_clip_0to1':
            self._calculate_normalization_params_custom_clip_0to1(xys_2d)
        elif norm_type == 'clip_minus1to1':
            ub = 1.0
            lb = -1.0
            fmins = np.min(xys_2d, axis=0)
            fmaxs = np.max(xys_2d, axis=0)
            self.slopes = (ub - lb) / (fmaxs - fmins)
            self.intercepts = (lb*fmaxs - ub*fmins) / (fmaxs - fmins)
            self.norm_type = 'linear_rescale'
        elif norm_type == 'none':
            self.norm_type = 'none'
        else:
            assert False, 'Incorrect parameter norm type selected: {}' \
                .format(self.param_dict['norm_type'])

    def _calculate_normalization_params_clip_0to1(self, xys_2d):
        ub = 1.0
        lb = 0.0
        fmins = np.min(xys_2d, axis=0)
        fmaxs = np.max(xys_2d, axis=0)

        self.slopes = (ub - lb) / (fmaxs - fmins)
        self.intercepts = (lb * fmaxs - ub * fmins) / (fmaxs - fmins)
        self.norm_type = 'linear_rescale'

    def _calculate_normalization_params_custom_clip_0to1(self, xys_2d):
        # linearly map the range specified to [0, 1]; if unspecified, use clip_0to1
        ub = 1.0
        lb = 0.0
        fmins = np.min(xys_2d, axis=0)
        fmaxs = np.max(xys_2d, axis=0)

        if 'custom_clip_0to1_map' in self.param_dict:
            custom_map = self.param_dict['custom_clip_0to1_map']
            features = self.model_dict['feature_names']
            for feature in custom_map:
                if feature in features:
                    fmin, fmax = custom_map[feature]
                    idx = features.index(feature)
                    assert len(fmins) == len(features) + 1 # fmins[0] is for y
                    assert len(fmins) == len(features) + 1 # fmaxs[0] is for y
                    fmins[idx + 1] = fmin
                    fmaxs[idx + 1] = fmax

        self.slopes = (ub - lb) / (fmaxs - fmins)
        self.intercepts = (lb * fmaxs - ub * fmins) / (fmaxs - fmins)
        self.norm_type = 'linear_rescale'

    def _normalize_xys(self, xys_2d):
        if self.norm_type == 'linear_rescale':
            xys_2d = self.slopes * xys_2d + self.intercepts
        elif self.norm_type == 'none':
            pass
        else:
            assert False, 'Incorrect model norm type selected: {}' \
                .format(self.norm_type)
        return xys_2d

    def denormalize_ys(self, ys_vec):
        if self.norm_type == 'linear_rescale':
            ys_vec = (ys_vec - self.intercepts[0]) / self.slopes[0]
        elif self.norm_type == 'none':
            pass
        else:
            assert False, 'Incorrect model norm type selected: {}' \
                .format(self.norm_type)
        return ys_vec

    def normalize_xs(self, xs_2d):
        if self.norm_type == 'linear_rescale':
            xs_2d = self.slopes[1:] * xs_2d + self.intercepts[1:]
        elif self.norm_type == 'none':
            pass
        else:
            assert False, 'Incorrect model norm type selected: {}' \
                .format(self.norm_type)
        return xs_2d

    def _preproc_predict(self, xs):
        self._assert_trained()
        feature_names = self.feature_names
        for name in feature_names:
            assert name in xs
        xs_2d = self._to_tabular_xs(feature_names, xs)
        # normalize xs
        xs_2d = self.normalize_xs(xs_2d)
        return xs_2d

    def predict(self, xs):
        xs_2d = self._preproc_predict(xs)
        ys_label_pred = self._predict(self.model, xs_2d)
        ys_label_pred = self.denormalize_ys(ys_label_pred)
        return {'ys_label_pred': ys_label_pred}

    @classmethod
    def _to_tabular_xys(cls, xkeys, xys):
        xs_2d = None
        for name in xkeys:
            if xs_2d is None:
                xs_2d = np.matrix(xys[name]).T
            else:
                xs_2d = np.hstack((xs_2d, np.matrix(xys[name]).T))

        # combine them
        ys_vec = xys['label']
        xys_2d = np.array(np.hstack((np.matrix(ys_vec).T, xs_2d)))
        return xys_2d

    @classmethod
    def _to_tabular_xs(cls, xkeys, xs):
        xs_2d = []
        for name in xkeys:
            xs_2d.append(np.array(xs[name]))
        xs_2d = np.vstack(xs_2d).T
        return xs_2d

    def evaluate(self, xs, ys):
        ys_label_pred = self.predict(xs)['ys_label_pred']
        ys_label = ys['label']
        stats = self.get_stats(ys_label, ys_label_pred)
        return stats

    @classmethod
    def delete(cls, filename):
        cls._delete(filename)

    @staticmethod
    def _delete(filename):
        if os.path.exists(filename):
            os.remove(filename)

    @classmethod
    def get_xs_from_results(cls, results, indexs=None, aggregate=True):
        """
        :param results: list of BasicResult, or pandas.DataFrame
        :param indexs: indices of results to be used
        :param aggregate: if True, return aggregate score, otherwise per-frame/per-block
        """
        try:
            if aggregate:
                feature_names = results[0].get_ordered_list_score_key()
            else:
                feature_names = results[0].get_ordered_list_scores_key()
        except AttributeError:
            # if RawResult, will not have either get_ordered_list_score_key
            # or get_ordered_list_scores_key. Instead, just get the sorted keys
            feature_names = results[0].get_ordered_results()

        feature_names = to_list(feature_names)
        cls._assert_dimension(feature_names, results)

        # collect results into xs
        xs = {}
        for name in feature_names:
            if indexs is not None:
                _results = to_list(map(lambda i:results[i], indexs))
            else:
                _results = results
            xs[name] = to_list(map(lambda result: result[name], _results))
        return xs

    @classmethod
    def _assert_dimension(cls, feature_names, results):
        # by default, only accept result[feature_name] that is a scalar
        for name in feature_names:
            for result in results:
                assert isinstance(result[name], Number)

    @staticmethod
    def get_per_unit_xs_from_a_result(result):
        """
        Similar to get_xs_from_results(), except that instead of intake a list
        of Result, each corresponding to an aggregate score, this function takes
        a single Result, and interpret its per-frame score as an aggregate score.
        :param result: one BasicResult
        """
        # need to substitute the score key (e.g. motion_score -> motion_scores)
        # to ensure compatibility
        feature_names = result.get_ordered_list_scores_key()
        new_feature_names = result.get_ordered_list_score_key()
        xs = {}
        for name, new_name in zip(feature_names, new_feature_names):
            xs[new_name] = np.array(result[name])
        return xs

    @classmethod
    def get_ys_from_results(cls, results, indexs=None):
        """
        :param results: list of BasicResult, or pandas.DataFrame
        :param indexs: indices of results to be used
        """
        ys = {}
        if indexs is not None:
            _results = to_list(map(lambda i:results[i], indexs))
        else:
            _results = results
        ys['label'] = \
            np.array(to_list(map(lambda result: result.asset.groundtruth, _results)))
        ys['content_id'] = \
            np.array(to_list(map(lambda result: result.asset.content_id, _results)))
        return ys

    @classmethod
    def get_xys_from_results(cls, results, indexs=None, aggregate=True):
        """
        :param results: list of BasicResult, or pandas.DataFrame
        :param indexs: indices of results to be used
        """
        xys = {}
        xys.update(cls.get_xs_from_results(results, indexs, aggregate))
        xys.update(cls.get_ys_from_results(results, indexs))
        return xys

    @classmethod
    def reset(cls):
        # placeholder for adding any reset mechanism to avoid interference
        # between experiments
        pass


class LibsvmNusvrTrainTestModel(TrainTestModel, RegressorMixin):

    TYPE = 'LIBSVMNUSVR'
    VERSION = "0.1"

    @classmethod
    def _train(cls, model_param, xys_2d, **kwargs):
        """
        :param model_param:
        :param xys_2d:
        :return:
        """
        kernel = model_param['kernel'] if 'kernel' in model_param else 'rbf'
        gamma = model_param['gamma'] if 'gamma' in model_param else 0.0
        C = model_param['C'] if 'C' in model_param else 1.0
        nu = model_param['nu'] if 'nu' in model_param else 0.5
        cache_size = model_param['cache_size'] if 'cache_size' in model_param else 200

        try:
            svmutil
        except NameError:
            from vmaf import svmutil

        if kernel == 'rbf':
            ktype_int = svmutil.RBF
        elif kernel == 'linear':
            ktype_int = svmutil.LINEAR
        elif kernel == 'poly':
            ktype_int = svmutil.POLY
        elif kernel == 'sigmoid':
            ktype_int = svmutil.SIGMOID
        else:
            assert False, 'ktype = ' + str(kernel) + ' not implemented'

        param = svmutil.svm_parameter([
            '-s', 4,
            '-t', ktype_int,
            '-c', C,
            '-g', gamma,
            '-n', nu,
            '-m', cache_size
        ])

        f = list(xys_2d[:, 1:])
        for i, item in enumerate(f):
            f[i] = list(item)
        prob = svmutil.svm_problem(xys_2d[:, 0], f)
        model = svmutil.svm_train(prob, param)

        return model

    @classmethod
    def _predict(cls, model, xs_2d):
        # override TrainTestModel._predict
        try:
            svmutil
        except NameError:
            from vmaf import svmutil

        f = list(xs_2d)
        for i, item in enumerate(f):
            f[i] = list(item)
        score, _, _ = svmutil.svm_predict([0] * len(f), f, model)
        ys_label_pred = np.array(score)
        return ys_label_pred

    @staticmethod
    def _to_file(filename, param_dict, model_dict):
        try:
            svmutil
        except NameError:
            from vmaf import svmutil

        # override TrainTestModel._to_file
        # special handling of libsvmnusvr: save .model differently
        info_to_save = {'param_dict': param_dict,
                        'model_dict': model_dict.copy()}
        svm_model = info_to_save['model_dict']['model']
        info_to_save['model_dict']['model'] = None
        with open(filename, 'wb') as file:
            pickle.dump(info_to_save, file)
        svmutil.svm_save_model(filename + '.model', svm_model)

    @classmethod
    def _from_info_loaded(cls, info_loaded, filename, logger, optional_dict2):
        try:
            svmutil
        except NameError:
            from vmaf import svmutil

        # override TrainTestModel._from_info_loaded
        train_test_model = cls(
            param_dict={}, logger=logger, optional_dict2=optional_dict2)
        train_test_model.param_dict = info_loaded['param_dict']
        train_test_model.model_dict = info_loaded['model_dict']

        if issubclass(cls, LibsvmNusvrTrainTestModel):
            # == special handling of libsvmnusvr: load .model differently ==
            model = svmutil.svm_load_model(filename + '.model')
            train_test_model.model_dict['model'] = model

        return train_test_model

    @classmethod
    def _delete(cls, filename):
        # override TrainTestModel._delete
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(filename + '.model'):
            os.remove(filename + '.model')

    @classmethod
    def from_raw_file(cls, model_filename, additional_model_dict, logger):
        """
        Construct from raw libsvm model file.
        :param model_filename:
        :param additional_model_dict: must contain keys feature_names, norm_type
        and optional slopes and intercepts
        :param logger:
        :return:
        """
        try:
            svmutil
        except NameError:
            from vmaf import svmutil

        # assert additional_model_dict
        assert 'feature_names' in additional_model_dict
        assert 'norm_type' in additional_model_dict
        norm_type = additional_model_dict['norm_type']
        assert norm_type == 'none' or norm_type == 'linear_rescale'
        if norm_type == 'linear_rescale':
            assert 'slopes' in additional_model_dict
            assert 'intercepts' in additional_model_dict

        train_test_model = cls(param_dict={}, logger=logger)

        train_test_model.model_dict.update(additional_model_dict)

        model = svmutil.svm_load_model(model_filename)
        train_test_model.model_dict['model'] = model

        return train_test_model


class SklearnRandomForestTrainTestModel(TrainTestModel, RegressorMixin):

    TYPE = 'RANDOMFOREST'
    VERSION = "0.1"

    @classmethod
    def _train(cls, model_param, xys_2d, **kwargs):
        """
        random forest regression
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        :param model_param:
        :param xys_2d:
        :return:
        """
        model_param_ = model_param.copy()

        # remove keys unassociated with sklearn
        if 'norm_type' in model_param_:
            del model_param_['norm_type']
        if 'score_clip' in model_param_:
            del model_param_['score_clip']
        if 'custom_clip_0to1_map' in model_param_:
            del model_param_['custom_clip_0to1_map']
        if 'num_models' in model_param_:
            del model_param_['num_models']

        from sklearn import ensemble
        model = ensemble.RandomForestRegressor(
            **model_param_
        )
        model.fit(xys_2d[:, 1:], np.ravel(xys_2d[:, 0]))

        return model

    @classmethod
    def _predict(cls, model, xs_2d):
        # directly call sklearn's model's predict() function
        ys_label_pred = model.predict(xs_2d)
        return ys_label_pred


class SklearnLinearRegressionTrainTestModel(TrainTestModel, RegressorMixin):

    TYPE = 'LINEARREG'
    VERSION = "0.1"

    @classmethod
    def _train(cls, model_param, xys_2d, **kwargs):
        """
        linear regression
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        :param model_param:
        :param xys_2d:
        :return:
        """
        model_param_ = model_param.copy()

        # remove keys unassociated with sklearn
        if 'norm_type' in model_param_:
            del model_param_['norm_type']
        if 'score_clip' in model_param_:
            del model_param_['score_clip']
        if 'custom_clip_0to1_map' in model_param_:
            del model_param_['custom_clip_0to1_map']
        if 'num_models' in model_param_:
            del model_param_['num_models']

        from sklearn import linear_model
        model = linear_model.LinearRegression(
            **model_param_
        )
        model.fit(xys_2d[:, 1:], np.ravel(xys_2d[:, 0]))

        return model

    @classmethod
    def _predict(cls, model, xs_2d):
        # directly call sklearn's model's predict() function
        ys_label_pred = model.predict(xs_2d)
        return ys_label_pred


class SklearnExtraTreesTrainTestModel(TrainTestModel, RegressorMixin):

    TYPE = 'EXTRATREES'
    VERSION = "0.1"

    @classmethod
    def _train(cls, model_param, xys_2d, **kwargs):
        """
        extremely random trees
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
        :param model_param:
        :param xys_2d:
        :return:
        """
        model_param_ = model_param.copy()

        # remove keys unassociated with sklearn
        if 'norm_type' in model_param_:
            del model_param_['norm_type']
        if 'score_clip' in model_param_:
            del model_param_['score_clip']
        if 'custom_clip_0to1_map' in model_param_:
            del model_param_['custom_clip_0to1_map']
        if 'num_models' in model_param_:
            del model_param_['num_models']

        from sklearn import ensemble
        model = ensemble.ExtraTreesRegressor(
            **model_param_
        )
        model.fit(xys_2d[:, 1:], np.ravel(xys_2d[:, 0]))

        return model

    @classmethod
    def _predict(cls, model, xs_2d):
        # directly call sklearn's model's predict() function
        ys_label_pred = model.predict(xs_2d)
        return ys_label_pred


class RawVideoTrainTestModelMixin(object):
    """
    Contains some key methods to handle input being RawVideoExtractor results.
    """

    @classmethod
    def _assert_dimension(cls, feature_names, results):
        # Override TrainTestModel._assert_dimension. Allow input to be a numpy
        # ndarray or equivalent (e.g. H5py object) -- they must have attribute
        # 'shape', and the shape must be 3-dimensional (frames, height, width)
        assert hasattr(results[0][feature_names[0]], 'shape')
        for result in results:
            for feature_name in feature_names:
                # esult[feature_name] is video of dims: frames, height, width
                assert len(result[feature_name].shape) == 3


class MomentRandomForestTrainTestModel(RawVideoTrainTestModelMixin,
                                       # : order affects whose _assert_dimension
                                       # gets called
                                       SklearnRandomForestTrainTestModel,
                                       RegressorMixin,
                                       ):
    """
    Compute moments based on the input videos (each video of form frames x
     width x height 3D-array) and then call a RandomForestTrainTestModel.
     For demo purpose only.
    """

    TYPE = 'MOMENTRANDOMFOREST'
    VERSION = "0.1"

    @classmethod
    def _to_tabular_xys(cls, xkeys, xys):
        # Override TrainTestModel._to_tabular_xys. For each image, extract
        # 1st, 2nd moment and var

        # get xs first
        xs_2d = cls._to_tabular_xs(xkeys, xys)

        # combine with ys
        ys_vec = xys['label']
        xys_2d = np.array(np.hstack((np.matrix(ys_vec).T, xs_2d)))

        return xys_2d

    @classmethod
    def _to_tabular_xs(cls, xkeys, xs):
        # Override TrainTestModel._to_tabular_xs
        # from raw video to 1st, 2nd moment and var, format xs properly in
        # tabular form
        sorted_xkeys = sorted(xkeys)

        xs_list = []
        for key in sorted_xkeys:
            videos = xs[key]
            video_stats_list = []
            for video in videos:
                nframes = video.shape[0]
                frame_stats = np.zeros((nframes, 3))
                for iframe, frame in enumerate(video):
                    firstm = frame.mean()
                    variance = frame.var()
                    secondm = variance + firstm**2
                    frame_stats[iframe] = (firstm, secondm, variance)
                video_stats = np.mean(frame_stats, axis=0)
                video_stats_list.append(video_stats)
            video_stats_2d = np.vstack(video_stats_list)
            xs_list.append(video_stats_2d)
        xs_2d = np.hstack(xs_list)

        return xs_2d


class BootstrapRegressorMixin(RegressorMixin):

    @classmethod
    def get_stats(cls, ys_label, ys_label_pred, **kwargs):
        # override RegressionMixin.get_stats
        try:
            assert 'ys_label_pred_bagging' in kwargs
            assert 'ys_label_pred_stddev' in kwargs
            assert 'ys_label_pred_ci95_low' in kwargs
            assert 'ys_label_pred_ci95_high' in kwargs
            stats = super(BootstrapRegressorMixin, cls).get_stats(ys_label, ys_label_pred, **kwargs)
            stats['ys_label_pred_bagging'] = kwargs['ys_label_pred_bagging']
            stats['ys_label_pred_stddev'] = kwargs['ys_label_pred_stddev']
            stats['ys_label_pred_ci95_low'] = kwargs['ys_label_pred_ci95_low']
            stats['ys_label_pred_ci95_high'] = kwargs['ys_label_pred_ci95_high']
            return stats
        except AssertionError:
            return super(BootstrapRegressorMixin, cls).get_stats(ys_label, ys_label_pred, **kwargs)

    @classmethod
    def plot_scatter(cls, ax, stats, **kwargs):
        # override RegressionMixin.plot_scatter

        assert len(stats['ys_label']) == len(stats['ys_label_pred'])

        content_ids = kwargs['content_ids'] if 'content_ids' in kwargs else None
        point_labels = kwargs['point_labels'] if 'point_labels' in kwargs else None

        try:

            ci_assume_gaussian = kwargs['ci_assume_gaussian'] if 'ci_assume_gaussian' in kwargs else True

            assert 'ys_label_pred_bagging' in stats
            assert 'ys_label_pred_stddev' in stats
            assert 'ys_label_pred_ci95_low' in stats
            assert 'ys_label_pred_ci95_high' in stats
            avg_std = np.mean(stats['ys_label_pred_stddev'])
            avg_ci95_low = np.mean(stats['ys_label_pred_ci95_low'])
            avg_ci95_high = np.mean(stats['ys_label_pred_ci95_high'])
            if content_ids is None:
                if ci_assume_gaussian:
                    yerr = 1.96 * stats['ys_label_pred_stddev'] # 95% C.I. (assume Gaussian)
                else:
                    yerr = [stats['ys_label_pred_bagging'] - avg_ci95_low, avg_ci95_high - stats['ys_label_pred_bagging']] # 95% C.I.
                ax.errorbar(stats['ys_label'], stats['ys_label_pred'],
                            yerr=yerr,
                            marker='o', linestyle='')
            else:
                assert len(stats['ys_label']) == len(content_ids)

                unique_content_ids = list(set(content_ids))
                from vmaf import plt
                cmap = plt.get_cmap()
                colors = [cmap(i) for i in np.linspace(0, 1, len(unique_content_ids))]
                for idx, curr_content_id in enumerate(unique_content_ids):
                    curr_idxs = indices(content_ids, lambda cid: cid == curr_content_id)
                    curr_ys_label = np.array(stats['ys_label'])[curr_idxs]
                    curr_ys_label_pred = np.array(stats['ys_label_pred'])[curr_idxs]
                    curr_ys_label_pred_bagging = np.array(stats['ys_label_pred_bagging'])[curr_idxs]
                    curr_ys_label_pred_stddev = np.array(stats['ys_label_pred_stddev'])[curr_idxs]
                    curr_ys_label_pred_ci95_low = np.array(stats['ys_label_pred_ci95_low'])[curr_idxs]
                    curr_ys_label_pred_ci95_high = np.array(stats['ys_label_pred_ci95_high'])[curr_idxs]
                    if ci_assume_gaussian:
                        yerr = 1.96 * curr_ys_label_pred_stddev # 95% C.I. (assume Gaussian)
                    else:
                        yerr = [curr_ys_label_pred_bagging - curr_ys_label_pred_ci95_low, curr_ys_label_pred_ci95_high - curr_ys_label_pred_bagging] # 95% C.I.
                    try:
                        curr_ys_label_stddev = np.array(stats['ys_label_stddev'])[curr_idxs]
                        ax.errorbar(curr_ys_label, curr_ys_label_pred,
                                    yerr=yerr,
                                    xerr=1.96 * curr_ys_label_stddev,
                                    marker='o', linestyle='', label=curr_content_id, color=colors[idx % len(colors)])
                    except:
                        ax.errorbar(curr_ys_label, curr_ys_label_pred,
                                    yerr=yerr,
                                    marker='o', linestyle='', label=curr_content_id, color=colors[idx % len(colors)])

            ax.text(0.45, 0.1, 'Avg. Pred. Std.: {:.2f}'.format(avg_std),
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    fontsize=12)

            if point_labels:
                assert len(point_labels) == len(stats['ys_label'])
                for i, point_label in enumerate(point_labels):
                    ax.annotate(point_label, (stats['ys_label'][i], stats['ys_label_pred'][i]))

        except AssertionError:
            super(BootstrapRegressorMixin, cls).plot_scatter(ax, stats, **kwargs)


class BootstrapMixin(object):

    MIXIN_VERSION = 'B0.0.1'

    # since bootstrap version 0.6.3, we want num_models to be + 1 the number of bootstrap model.
    # Therefore, we use 100 bootstrap models or (DEFAULT_NUM_MODELS - 1) bootstrap models.
    DEFAULT_NUM_MODELS = 101

    def train(self, xys, **kwargs):
        # override TrainTestModel.train()
        xys_2d = self._preproc_train(xys)
        num_models = self._get_num_models()
        sample_size = xys_2d.shape[0]
        models = []

        # first model: use full training data
        model_0 = self._train(self.param_dict, xys_2d, **kwargs)
        models.append(model_0)

        # rest models: resample training data with replacement
        for i_model in range(1, num_models):
            np.random.seed(i_model) # seed is i_model
            # random sample with replacement:
            indices = np.random.choice(range(sample_size), size=sample_size, replace=True)
            xys_2d_ = xys_2d[indices, :]
            model_ = self._train(self.param_dict, xys_2d_, **kwargs)
            models.append(model_)
        self.model = models

    def _get_num_models(self):
        num_models = self.param_dict[
            'num_models'] if 'num_models' in self.param_dict else self.DEFAULT_NUM_MODELS
        return num_models

    @classmethod
    def _get_num_models_from_param_dict(cls, param_dict):
        num_models = param_dict[
            'num_models'] if 'num_models' in param_dict else cls.DEFAULT_NUM_MODELS
        return num_models

    def predict(self, xs):
        # override TrainTestModel.predict()
        xs_2d = self._preproc_predict(xs)

        models = self.model
        num_models = self._get_num_models()
        assert num_models == len(models)

        # first model: conventional prediction
        model_0 = models[0]
        ys_label_pred = self._predict(model_0, xs_2d)
        ys_label_pred = self.denormalize_ys(ys_label_pred)

        # rest models: bagging (bootstrap aggregation)
        # first check if there are any bootstrapped models
        if num_models > 1:
            ys_list = []
            for model_ in models[1:]:
                ys = self._predict(model_, xs_2d)
                ys_list.append(ys)
            ys_2d = np.vstack(ys_list)
            ys_2d = self.denormalize_ys(ys_2d)
            ys_label_pred_bagging = np.mean(ys_2d, axis=0)
            ys_label_pred_stddev = np.std(ys_2d, axis=0)
            ys_label_pred_ci95_low = np.percentile(ys_2d, 2.5, axis=0)
            ys_label_pred_ci95_high = np.percentile(ys_2d, 97.5, axis=0)
            return {'ys_label_pred_all_models': ys_2d,
                    'ys_label_pred': ys_label_pred,
                    'ys_label_pred_bagging': ys_label_pred_bagging,
                    'ys_label_pred_stddev': ys_label_pred_stddev,
                    'ys_label_pred_ci95_low': ys_label_pred_ci95_low,
                    'ys_label_pred_ci95_high': ys_label_pred_ci95_high,
                    }
        else:
            return {'ys_label_pred': ys_label_pred,
                    }

    def evaluate_stddev(self, xs):
        prediction = self.predict(xs)
        return {'mean_stddev': np.mean(prediction['ys_label_pred_stddev']),
                'mean_ci95_low': np.mean(prediction['ys_label_pred_ci95_low']),
                'mean_ci95_high': np.mean(prediction['ys_label_pred_ci95_high'])}

    def evaluate_bagging(self, xs, ys):
        ys_label_pred_bagging = self.predict(xs)['ys_label_pred_bagging']
        ys_label = ys['label']
        stats = self.get_stats(ys_label, ys_label_pred_bagging)
        return stats

    def to_file(self, filename):
        # override TrainTestModel.to_file()
        self._assert_trained()
        param_dict = self.param_dict
        model_dict = self.model_dict

        models = self.model
        num_models = self._get_num_models()
        assert num_models == len(models)
        for i_model, model in enumerate(models):
            filename_ = self._get_model_i_filename(filename, i_model)
            filedir = os.path.dirname(filename_)
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            model_dict_ = model_dict.copy()
            model_dict_['model'] = model
            self._to_file(filename_, param_dict, model_dict_)

    @staticmethod
    def _get_model_i_filename(filename, i_model):
        # first model doesn't have suffix - so it have the same file name as a regular model
        if i_model == 0:
            filename_ = "{}".format(filename)
        else:
            filename_ = "{}.{:04d}".format(filename, i_model)
        return filename_

    @classmethod
    def from_file(cls, filename, logger=None, optional_dict2=None):
        # override TrainTestModel.from_file()
        filename_0 = cls._get_model_i_filename(filename, 0)
        assert os.path.exists(filename_0), 'File name {} does not exist.'.format(filename_0)
        with open(filename_0, 'rb') as file:
            info_loaded_0 = pickle.load(file)
        model_type = info_loaded_0['model_dict']['model_type']
        model_class = TrainTestModel.find_subclass(model_type)
        train_test_model_0 = model_class._from_info_loaded(
            info_loaded_0, filename_0, logger, optional_dict2)
        num_models = cls._get_num_models_from_param_dict(info_loaded_0['param_dict'])

        models = []
        for i_model in range(num_models):
            filename_ = cls._get_model_i_filename(filename, i_model)
            assert os.path.exists(filename_), 'File name {} does not exist.'.format(filename_)
            with open(filename_, 'rb') as file:
                info_loaded_ = pickle.load(file)
            train_test_model_ = model_class._from_info_loaded(info_loaded_, filename_, None, None)
            model_ = train_test_model_.model
            models.append(model_)

        train_test_model_0.model = models

        return train_test_model_0

    @classmethod
    def delete(cls, filename):
        # override TrainTestModel.delete()
        filename_0 = cls._get_model_i_filename(filename, 0)
        assert os.path.exists(filename_0)
        with open(filename_0, 'rb') as file:
            info_loaded_0 = pickle.load(file)
        num_models = cls._get_num_models_from_param_dict(info_loaded_0['param_dict'])
        for i_model in range(num_models):
            filename_ = cls._get_model_i_filename(filename, i_model)
            cls._delete(filename_)


class BootstrapLibsvmNusvrTrainTestModel(BootstrapRegressorMixin, BootstrapMixin, LibsvmNusvrTrainTestModel):

    TYPE = 'BOOTSTRAP_LIBSVMNUSVR'
    VERSION = LibsvmNusvrTrainTestModel.VERSION + '-' + BootstrapMixin.MIXIN_VERSION


class BootstrapSklearnRandomForestTrainTestModel(BootstrapRegressorMixin, BootstrapMixin, SklearnRandomForestTrainTestModel):

    TYPE = 'BOOTSTRAP_RANDOMFOREST'
    VERSION = SklearnRandomForestTrainTestModel.VERSION + '-' + BootstrapMixin.MIXIN_VERSION


class ResidueBootstrapMixin(BootstrapMixin):

    MIXIN_VERSION = 'RB0.0.1'

    def train(self, xys, **kwargs):
        # override TrainTestModel.train()
        xys_2d = self._preproc_train(xys)
        num_models = self._get_num_models()
        sample_size = xys_2d.shape[0]
        models = []

        # first model: use full training data
        model_0 = self._train(self.param_dict, xys_2d, **kwargs)
        models.append(model_0)

        # predict and find residue
        ys = xys_2d[:, 0].T
        xs_2d = xys_2d[:, 1:]
        ys_pred = self._predict(model_0, xs_2d)
        residue_ys = ys - ys_pred

        # rest models: resample residue data with replacement
        for i_model in range(1, num_models):
            np.random.seed(i_model) # seed is i_model
            # random sample with replacement:
            indices = np.random.choice(range(sample_size), size=sample_size, replace=True)
            residue_ys_resampled = residue_ys[indices]
            ys_resampled = residue_ys_resampled + ys_pred
            xys_2d_ = np.array(np.hstack((np.matrix(ys_resampled).T, xs_2d)))
            model_ = self._train(self.param_dict, xys_2d_, **kwargs)
            models.append(model_)
        self.model = models


class ResidueBootstrapLibsvmNusvrTrainTestModel(BootstrapRegressorMixin, ResidueBootstrapMixin, LibsvmNusvrTrainTestModel):

    TYPE = 'RESIDUEBOOTSTRAP_LIBSVMNUSVR'
    VERSION = LibsvmNusvrTrainTestModel.VERSION + '-' + ResidueBootstrapMixin.MIXIN_VERSION


class ResidueBootstrapRandomForestTrainTestModel(BootstrapRegressorMixin, ResidueBootstrapMixin, SklearnRandomForestTrainTestModel):

    TYPE = 'RESIDUEBOOTSTRAP_RANDOMFOREST'
    VERSION = SklearnRandomForestTrainTestModel.VERSION + '-' + ResidueBootstrapMixin.MIXIN_VERSION