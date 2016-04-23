__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import sys
import pickle

import scipy.stats
import numpy as np
from numpy.linalg import lstsq
import pandas as pd

import config
from tools.misc import indices
from core.mixin import TypeVersionEnabled


class TrainTestModel(TypeVersionEnabled):

    def __init__(self, param_dict, logger=None):
        """
        :param param_dict: contains input parameters
        :param logger:
        :return:
        """
        TypeVersionEnabled.__init__(self)

        self.param_dict = param_dict
        self.logger = logger

        self.model_dict = {}

    @property
    def model_id(self):
        return TypeVersionEnabled.get_type_version_string(self)

    def _assert_trained(self):

        assert 'model_type' in self.model_dict # need this to recover class

        assert 'feature_names' in self.model_dict

        assert 'model' in self.model_dict

        assert 'norm_type' in self.model_dict

        norm_type = self.model_dict['norm_type']
        assert (   norm_type == 'none'
                or norm_type == 'linear_rescale')

        if norm_type == 'linear_rescale':
            assert 'slopes' in self.model_dict
            assert 'intercepts' in self.model_dict

    def append_info(self, key, value):
        """
        Useful for adding extra info to model before saving. For example,
        save feature_dict to model so that when the model is loaded by a
        QualityRunner, it knows when features to extract.
        :param key:
        :param value:
        :return:
        """
        self.model_dict[key] = value

    def get_appended_info(self, key):
        """
        Retrieve info added via the append_info method.
        :param key:
        :return:
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
        return self.model_dict['mus']
    @mus.setter
    def mus(self, value):
        self.model_dict['mus'] = value

    @property
    def sds(self):
        return self.model_dict['sds']
    @sds.setter
    def sds(self, value):
        self.model_dict['sds'] = value

    @property
    def slopes(self):
        return self.model_dict['slopes']
    @slopes.setter
    def slopes(self, value):
        self.model_dict['slopes'] = value

    @property
    def intercepts(self):
        return self.model_dict['intercepts']
    @intercepts.setter
    def intercepts(self, value):
        self.model_dict['intercepts'] = value

    @property
    def model(self):
        return self.model_dict['model']
    @model.setter
    def model(self, value):
        self.model_dict['model'] = value

    def to_file(self, filename):

        self._assert_trained()
        info_to_save = {'param_dict': self.param_dict,
                        'model_dict': self.model_dict}

        with open(filename, 'wb') as file:
            pickle.dump(info_to_save, file)

    @staticmethod
    def from_file(filename, logger):

        with open(filename, 'rb') as file:
            info_loaded = pickle.load(file)

        model_type = info_loaded['model_dict']['model_type']

        if model_type == LibsvmnusvrTrainTestModel.TYPE:
            train_test_model = LibsvmnusvrTrainTestModel(param_dict={}, logger=logger)
            train_test_model.param_dict = info_loaded['param_dict']
            train_test_model.model_dict = info_loaded['model_dict']
            # == special handling of libsvmnusvr: load .model differently ==
            model = LibsvmnusvrTrainTestModel.svmutil.svm_load_model(filename + '.model')
            train_test_model.model_dict['model'] = model
        else:
            model_class = TrainTestModel.find_subclass(model_type)
            train_test_model = model_class(param_dict={}, logger=logger)
            train_test_model.param_dict = info_loaded['param_dict']
            train_test_model.model_dict = info_loaded['model_dict']

        return train_test_model

    @staticmethod
    def delete(filename):
        if os.path.exists(filename):
            os.remove(filename)

    @staticmethod
    def _predict(model, xs_2d):
        ys_label_pred = model.predict(xs_2d)
        return ys_label_pred

    def predict(self, xs):

        self._assert_trained()

        for name in self.feature_names:
            assert name in xs

        xs_2d = None
        for name in self.feature_names:
            if xs_2d is None:
                xs_2d = np.matrix(xs[name]).T
            else:
                xs_2d = np.hstack((xs_2d, np.matrix(xs[name]).T))
        xs_2d = np.array(xs_2d)

        # normalize xs
        xs_2d = self.normalize_xs(xs_2d)

        # predict
        ys_label_pred = self._predict(self.model, xs_2d)

        # denormalize ys
        ys_label_pred = self.denormalize_ys(ys_label_pred)

        return ys_label_pred

    @staticmethod
    def sigmoid_adjust(xs, ys):
        ys_max = np.max(ys) + 0.1
        ys_min = np.min(ys) - 0.1

        # normalize to [0, 1]
        ys = list((np.array(ys) - ys_min) / (ys_max - ys_min))

        zs = -np.log(1.0 / np.array(ys).T - 1.0)
        Y_mtx = np.matrix((np.ones(len(ys)), zs)).T
        x_vec = np.matrix(xs).T
        a_b = lstsq(Y_mtx, x_vec)[0]
        a = a_b.item(0)
        b = a_b.item(1)

        xs = 1.0 / (1.0 + np.exp(- (np.array(xs) - a) / b))

        # denormalize
        xs = xs * (ys_max - ys_min) + ys_min

        return xs

    @classmethod
    def get_stats(cls, ys_label, ys_label_pred):

        # do adustment with sigmoid function
        ys_label_pred_adjusted = cls.sigmoid_adjust(ys_label_pred, ys_label)

        # MSE
        rmse = np.sqrt(np.mean(
            np.power(np.array(ys_label) - np.array(ys_label_pred_adjusted), 2.0)))
        # spearman
        srcc, _ = scipy.stats.spearmanr(ys_label, ys_label_pred_adjusted)
        # pearson
        pcc, _ = scipy.stats.pearsonr(ys_label, ys_label_pred_adjusted)
        # kendall
        kendall, _ = scipy.stats.kendalltau(ys_label, ys_label_pred_adjusted)
        stats = { 'RMSE': rmse,
                  'SRCC': srcc,
                  'PCC': pcc,
                  'KENDALL': kendall,
                  'ys_label': list(ys_label),
                  'ys_label_pred': list(ys_label_pred)}
        return stats

    @staticmethod
    def format_stats(stats):
        return '(SRCC: {srcc:.3f}, PCC: {pcc:.3f}, RMSE: {rmse:.3f})'.format(
            srcc=stats['SRCC'], pcc=stats['PCC'], rmse=stats['RMSE'])

    @staticmethod
    def format_stats2(stats):
        return 'RMSE: {rmse:.3f}\nPCC: {pcc:.3f}\nSRCC: {srcc:.3f}'.format(
            srcc=stats['SRCC'], pcc=stats['PCC'], rmse=stats['RMSE'])

    @staticmethod
    def format_stats3(stats):
        return 'SRCC: {srcc:.3f}, PCC: {pcc:.3f}'.format(
            srcc=stats['SRCC'], pcc=stats['PCC'], rmse=stats['RMSE'])

    @classmethod
    def aggregate_stats_list(cls, stats_list):
        aggregate_ys_label = []
        aggregate_ys_label_pred = []
        for stats in stats_list:
            aggregate_ys_label += stats['ys_label']
            aggregate_ys_label_pred += stats['ys_label_pred']
        return cls.get_stats(aggregate_ys_label, aggregate_ys_label_pred)

    @staticmethod
    def plot_scatter(ax, stats, content_ids=None):
        assert len(stats['ys_label']) == len(stats['ys_label_pred'])

        if content_ids is None:
            ax.scatter(stats['ys_label'], stats['ys_label_pred'])
        else:
            assert len(stats['ys_label']) == len(content_ids)

            unique_content_ids = list(set(content_ids))
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap()
            colors = [cmap(i) for i in np.linspace(0, 1, len(unique_content_ids))]
            for idx, curr_content_id in enumerate(unique_content_ids):
                curr_idxs = indices(content_ids, lambda cid: cid==curr_content_id)
                curr_ys_label = np.array(stats['ys_label'])[curr_idxs]
                curr_ys_label_pred = np.array(stats['ys_label_pred'])[curr_idxs]
                ax.scatter(curr_ys_label, curr_ys_label_pred,
                           label=curr_content_id, color=colors[idx % len(colors)])

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
            assert False, 'Unknow type: {} for get_score().'.format(type)

    def evaluate(self, xs, ys):
        ys_label_pred = self.predict(xs)
        ys_label = ys['label']
        return self.get_stats(ys_label, ys_label_pred)

    def train(self, xys):

        self.model_type = self.TYPE

        assert 'label' in xys

        ys_vec = xys['label']

        # this makes sure the order of features are normalized, and each
        # dimension of xys_2d is consistent with feature_names
        feature_names = sorted(xys.keys())

        feature_names.remove('label')
        feature_names.remove('content_id')

        # temp
        # feature_names = ['VMAF_feature_vif_score', 'VMAF_feature_adm_score', 'VMAF_feature_ansnr_score', 'VMAF_feature_motion_score']

        self.feature_names = feature_names

        xs_2d = []
        for name in feature_names:
            if xs_2d == []:
                xs_2d = np.matrix(xys[name]).T
            else:
                xs_2d = np.hstack((xs_2d, np.matrix(xys[name]).T))

        # combine them
        xys_2d = np.array(np.hstack((np.matrix(ys_vec).T, xs_2d)))

        # calculate normalization parameters,
        self._calculate_normalization_params(xys_2d)

        # normalize
        xys_2d = self._normalize_xys(xys_2d)

        model = self._train(self.param_dict, xys_2d)

        self.model = model

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
            ub = 1.0
            lb = 0.0
            fmins = np.min(xys_2d, axis=0)
            fmaxs = np.max(xys_2d, axis=0)
            self.slopes = (ub - lb) / (fmaxs - fmins)
            self.intercepts = (lb*fmaxs - ub*fmins) / (fmaxs - fmins)
            self.norm_type = 'linear_rescale'
        elif norm_type == 'clip_minus1to1':
            ub =  1.0
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

    @classmethod
    def get_xs_from_results(cls, results_or_df, indexs=None):
        """
        :param results_or_df: list of BasicResult, or pandas.DataFrame
        :param indexs: indices of results to be used
        :return:
        """

        if results_or_df.__class__ == pd.DataFrame:
            return cls.get_xs_from_dataframe(df=results_or_df, rows=indexs)

        feature_names = results_or_df[0].get_ordered_list_score_key()
        xs = {}
        for name in feature_names:
            _results = map(lambda i:results_or_df[i], indexs) \
                if indexs is not None \
                else results_or_df
            xs[name] = np.array(map(lambda result: result[name], _results))
        return xs

    @staticmethod
    def get_perframe_xs_from_result(result):
        """
        Similar to get_xs_from_results(), except that instead of intake a list
        of Result, each corresponding to an aggregate score, this function takes
        a single Result, and interpret its per-frame score as an aggregate score.
        :param result: one BasicResult
        :param indexs: indices of results to be used
        :return:
        """
        feature_names = result._get_ordered_list_scores_key()
        new_feature_names = result.get_ordered_list_score_key()
        xs = {}
        for name, new_name in zip(feature_names, new_feature_names):
            xs[new_name] = np.array(result[name])
        return xs

    @classmethod
    def get_ys_from_results(cls, results_or_df, indexs=None):
        """
        :param results_or_df: list of BasicResult, or pandas.DataFrame
        :param indexs: indices of results to be used
        :return:
        """
        if results_or_df.__class__ == pd.DataFrame:
            return cls.get_ys_from_dataframe(df=results_or_df, rows=indexs)

        ys = {}
        _results = map(lambda i:results_or_df[i], indexs) \
            if indexs is not None \
            else results_or_df
        ys['label'] = \
            np.array(map(lambda result: result.asset.groundtruth, _results))
        ys['content_id'] = \
            np.array(map(lambda result: result.asset.content_id, _results))
        return ys

    @classmethod
    def get_xys_from_results(cls, results_or_df, indexs=None):
        """
        :param results_or_df: list of BasicResult, or pandas.DataFrame
        :param indexs: indices of results to be used
        :return:
        """
        if results_or_df.__class__ == pd.DataFrame:
            return cls.get_xys_from_dataframe(df=results_or_df, rows=indexs)

        xys = {}
        xys.update(cls.get_xs_from_results(results_or_df, indexs))
        xys.update(cls.get_ys_from_results(results_or_df, indexs))
        return xys

    # ========================== begin of legacy ===============================

    # below is for the purpose of reading a legacy test text file, and to ensure
    # the code in the class produces bit-exact test results as before

    @staticmethod
    def get_xs_from_dataframe(df, rows=None):
        """Prepare xs (i.e. a dictionary of named features, e.g.
        xs = {'vif_feat': [0.8, 0.9, 0.5], 'ssim_feat': [1.0, 0.5, 0.6]}),
        which is to be used as input by predict(xs), from a pandas DataFrame
        df, e.g.
             ansnr_feat  content_id  distortion_id  ssim_feat     label
        0     0.8           0              0        1.0           8.4
        1     0.9           1              0        0.5           6.5
        0     0.5           0              0        0.6           4.3
        :param df:
        :param rows: if None, take all rows from df, otherwise must be a list of
        row indices
        :return:
        """
        # by the rule of Extraction, features always end with '_feat'
        feature_names = [name for name in df.columns.values if "_feat" in name]
        xs = {}
        for name in feature_names:
            if rows is None:
                xs[name] = np.array(df[name])
            else:
                xs[name] = np.array(df[name].iloc[rows])
        return xs

    @staticmethod
    def get_ys_from_dataframe(df, rows=None):
        """Prepare ys (i.e. a dictionary with key 'label' and labels, e.g.
        ys = {'label': [8.4, 6.5, 4.3]}), from a pandas DataFrame df, e.g.
             ansnr_feat  content_id  distortion_id  ssim_feat     label
        0     0.8           0              0        1.0           8.4
        1     0.9           1              0        0.5           6.5
        0     0.5           0              0        0.6           4.3
        :param df:
        :param rows: if None, take all rows from df, otherwise must be a list of
        row indices
        :return:
        """
        # by the rule of Extraction, labels must have key 'label'
        ys = {}
        if rows is None:
            ys['label'] = np.array(df['label'])
            ys['content_id'] = np.array(df['content_id'])
        else:
            ys['label'] = np.array(df['label'].iloc[rows])
            ys['content_id'] = np.array(df['content_id'].iloc[rows])
        return ys

    @classmethod
    def get_xys_from_dataframe(cls, df, rows=None):
        """Prepare xys (i.e. a dictionary of named features and labels, e.g.
        xys = {'vif_feat': [0.8, 0.9, 0.5], 'ssim_feat': [1.0, 0.5, 0.6],
        'label': [8.4, 6.5, 4.3]}), which is to be used as input by train(xys),
        from a pandas DataFrame df, e.g.
             ansnr_feat  content_id  distortion_id  ssim_feat     label
        0     0.8           0              0        1.0           8.4
        1     0.9           1              0        0.5           6.5
        0     0.5           0              0        0.6           4.3
        :param df:
        :param rows: if None, take all rows from df, otherwise must be a list of
        row indices
        :return:
        """
        xys = {}
        xys.update(cls.get_xs_from_dataframe(df, rows))
        xys.update(cls.get_ys_from_dataframe(df, rows))
        return xys

    # ========================== end of legacy =================================


class LibsvmnusvrTrainTestModel(TrainTestModel):

    TYPE = 'LIBSVMNUSVR'
    VERSION = "0.1"

    sys.path.append(config.ROOT + "/libsvm/python")
    import svmutil

    # override
    def to_file(self, filename):

        self._assert_trained()

        # special handling of libsvmnusvr: save .model differently
        model_dict_copy = self.model_dict.copy()
        model_dict_copy['model'] = None
        info_to_save = {'param_dict': self.param_dict,
                        'model_dict': model_dict_copy}
        self.svmutil.svm_save_model(filename + '.model', self.model_dict['model'])

        with open(filename, 'wb') as file:
            pickle.dump(info_to_save, file)

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

        # assert additional_model_dict
        assert 'feature_names' in additional_model_dict
        assert 'norm_type' in additional_model_dict
        norm_type = additional_model_dict['norm_type']
        assert (   norm_type == 'none'
                or norm_type == 'linear_rescale')
        if norm_type == 'linear_rescale':
            assert 'slopes' in additional_model_dict
            assert 'intercepts' in additional_model_dict

        train_test_model = cls(param_dict={}, logger=logger)

        train_test_model.model_dict.update(additional_model_dict)

        model = cls.svmutil.svm_load_model(model_filename)
        train_test_model.model_dict['model'] = model

        return train_test_model

    # override
    @staticmethod
    def delete(filename):
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(filename + '.model'):
            os.remove(filename + '.model')

    # override
    @classmethod
    def _predict(cls, model, xs_2d):
        f = list(xs_2d)
        for i, item in enumerate(f):
            f[i] = list(item)
        score, _, _ = cls.svmutil.svm_predict([0] * len(f), f, model)
        ys_label_pred = np.array(score)
        return ys_label_pred

    @classmethod
    def _train(cls, model_param, xys_2d):
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

        if kernel == 'rbf':
            ktype_int = cls.svmutil.RBF
        elif kernel == 'linear':
            ktype_int = cls.svmutil.LINEAR
        else:
            assert False, 'ktype = ' + str(kernel) + ' not implemented'

        param = cls.svmutil.svm_parameter(['-s', 4,
                                           '-t', ktype_int,
                                           '-c', C,
                                           '-g', gamma,
                                           '-n', nu,
                                           '-m', cache_size])

        f = list(xys_2d[:, 1:])
        for i, item in enumerate(f):
            f[i] = list(item)
        prob = cls.svmutil.svm_problem(xys_2d[:, 0], f)
        model = cls.svmutil.svm_train(prob, param)

        return model

class RandomForestTrainTestModel(TrainTestModel):

    TYPE = 'RANDOMFOREST'
    VERSION = "0.1"

    @staticmethod
    def _train(model_param, xys_2d):
        """
        random forest regression (interface scikit-learn 0.17.1
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        :param model_param:
        :param xys_2d:
        :return:
        """
        n_estimators = model_param['n_estimators'] if 'n_estimators' in model_param else 10
        criterion = model_param['criterion'] if 'criterion' in model_param else 'mse'
        max_depth = model_param['max_depth'] if 'max_depth' in model_param else None
        min_samples_split = model_param['min_samples_split'] if 'min_samples_split' in model_param else 2
        min_samples_leaf = model_param['min_samples_leaf'] if 'min_samples_leaf' in model_param else 1
        min_weight_fraction_leaf = model_param['min_weight_fraction_leaf'] if 'min_weight_fraction_leaf' in model_param else 0
        max_features = model_param['max_features'] if 'max_features' in model_param else 'auto'
        max_leaf_nodes = model_param['max_leaf_nodes'] if 'max_leaf_nodes' in model_param else None
        bootstrap = model_param['bootstrap'] if 'bootstrap' in model_param else True
        oob_score = model_param['oob_score'] if 'oob_score' in model_param else False
        n_jobs = model_param['n_jobs'] if 'n_jobs' in model_param else 1
        random_state = model_param['random_state'] if 'random_state' in model_param else None
        verbose = model_param['verbose'] if 'verbose' in model_param else 0
        warm_start = model_param['warm_start'] if 'warm_start' in model_param else False

        from sklearn import ensemble
        model = ensemble.RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )
        model.fit(xys_2d[:, 1:], np.ravel(xys_2d[:, 0]))

        return model

