__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import numpy as np
from tools import indices
import sys
import config

class TrainTestModel(object):

    def __init__(self, param_dict, logger=None):
        """
        :param param_dict: contains input parameters
        :param logger:
        :return:
        """
        self.param_dict = param_dict.copy() # keep original intact
        self.logger = logger

        self.norm_type = param_dict.setdefault('norm_type', 'whiten')
        self.mus = []
        self.sds = []
        self.fmins = []
        self.fmaxs = []
        self.model = None

        self.model_dict = {}

    # def _assert_trained(self):
    #
    #     # usually get from Result._get_ordered_list_score_key() except for
    #     # customly constructed
    #     assert 'feature_names' in self.model_dict
    #
    #     assert 'model' in self.model_dict
    #
    #     assert 'norm_type' in self.model_dict
    #     norm_type = self.model_dict['norm_type']
    #     assert (   norm_type == 'none'
    #             or norm_type == 'whiten'
    #             or norm_type == 'rescale_0to1'
    #             or norm_type == 'rescale_minus1to1')
    #
    #     if norm_type == 'normal':
    #         assert 'mu' in self.model_dict
    #         assert 'sd' in self.model_dict
    #
    #     if norm_type == 'clipped':
    #         assert 'fmin' in self.model_dict
    #         assert 'fmax' in self.model_dict

    # @property
    # def feature_names(self):
    #     self._assert_trained()
    #     return self.model_dict['feature_names']
    #
    # @feature_names.setter
    # def feature_names(self, value):
    #     self.model_dict['feature_names'] = value
    #
    # @property
    # def norm_type(self):
    #     self._assert_trained()
    #     return self.model_dict['norm_type']


    def to_file(self, filename):

        assert self.model is not None, "Nothing to save..."

        model_info = {}
        model_info['model'] = self.model
        model_info['params'] = self.param_dict
        model_info['mus'] = self.mus
        model_info['sds'] = self.sds
        model_info['fmins'] = self.fmins
        model_info['fmaxs'] = self.fmaxs
        model_info['norm_type'] = self.norm_type
        model_info['feature_names'] = self.feature_names

        import joblib
        joblib.dump(model_info, filename, compress=9)

        # self._assert_trained()
        # info_to_save = {}
        # info_to_save['param_dict'] = self.param_dict
        # info_to_save['model_dict'] = self.model_dict
        # import joblib
        # joblib.dump(info_to_save, filename, compress=9)

    @classmethod
    def from_file(cls, filename, logger):

        train_test_model = cls(param_dict={}, logger=logger)

        # import joblib
        # info_loaded = joblib.load(filename)
        # train_test_model.param_dict = info_loaded['param_dict']
        # train_test_model.model_dict = info_loaded['model_dict']

        import joblib
        model_info = joblib.load(filename)
        train_test_model.param_dict = model_info['params']
        train_test_model.model = model_info['model']
        train_test_model.mus = model_info['mus']
        train_test_model.sds = model_info['sds']
        train_test_model.fmins = model_info['fmins']
        train_test_model.fmaxs = model_info['fmaxs']
        # self.feature_names = model_info['feature_names']
        # for backward compatibility, use the following for older model files:
        train_test_model.feature_names = model_info['feature_names'] \
            if 'feature_names' in model_info else model_info['X_featurelabels']

        train_test_model.norm_type = model_info['norm_type']

        return train_test_model

    @staticmethod
    def _predict(model, xs_2d):
        ys_label_pred = model.predict(xs_2d)
        return ys_label_pred

    def predict(self, xs):

        # assert that xs is a dict
        expected_input = '''
           xs = {
                 'feature': [...],
                 'anotherfeature': [...],
                 'yetanotherfeature': [...],
           }
        '''
        assert type(xs) is dict, "predict only accepts named features passed " \
                                "via a dictionary! Expected input: \n" + \
                                expected_input

        for name in self.feature_names:
            assert name in xs, "Oops. " + name + " not found in feature dictionary."

        xs_2d = []
        for name in self.feature_names:
            if xs_2d == []:
                xs_2d = np.matrix(xs[name]).T
            else:
                xs_2d = np.hstack((xs_2d, np.matrix(xs[name]).T))
        xs_2d = np.array(xs_2d)

        # normalize xs
        xs_2d = self.normalize_xs(xs_2d)

        ys_label_pred = self._predict(self.model, xs_2d)

        # denormalize ys
        ys_label_pred = self.denormalize_ys(ys_label_pred)

        return ys_label_pred

    @staticmethod
    def get_stats(ys_label, ys_label_pred):

        import scipy.stats

        # MSE
        mse = np.mean(np.power(np.array(ys_label) - np.array(ys_label_pred), 2.0))

        # spearman
        srcc, _ = scipy.stats.spearmanr(ys_label, ys_label_pred)

        # pearson
        pcc, _ = scipy.stats.pearsonr(ys_label, ys_label_pred)

        # kendall
        kendall, _ = scipy.stats.kendalltau(ys_label, ys_label_pred)

        result = {}
        result['MSE'] = mse
        result['SRCC'] = srcc
        result['PCC'] = pcc
        result['KENDALL'] = kendall

        # append raw
        result['ys_label'] = list(ys_label)
        result['ys_label_pred'] = list(ys_label_pred)

        return result

    @classmethod
    def aggregate_stats_list(cls, results):
        aggregate_ys_label = []
        aggregate_ys_label_pred = []
        for result in results:
            aggregate_ys_label += result['ys_label']
            aggregate_ys_label_pred += result['ys_label_pred']
        return cls.get_stats(aggregate_ys_label, aggregate_ys_label_pred)

    @staticmethod
    def get_objective_score(result, type='SRCC'):
        """
        Objective score is something to MAXIMIZE. e.g. SRCC, or -MSE.
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
        elif type == 'MSE':
            return -result['MSE']
        else:
            assert False, 'Unknow type: {} for get_score().'.format(type)

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

    def evaluate(self, xs, ys):
        ys_label_pred = self.predict(xs)
        ys_label = ys['label']
        return self.get_stats(ys_label, ys_label_pred)

    def train(self, xys):
        # xys should be dictionary of vectors like
        expected_input = '''
           xys = { 'label': [...],
                 'content_id': [...],
                 'feature': [...],
                 'anotherfeature': [...],
                 'yetanotherfeature': [...],
           }
       '''
        # where you can add as many keys to X, but each should be set
        # with the vector of the same length as label
        assert type(xys) is dict, "train only accepts named features passed " \
                                "via a dictionary! Pass in something like:\n" \
                                + expected_input

        assert 'label' in xys, "Must pass in a label vector for training... " \
                             "i.e xys['label'] = [...] "

        ys_vec = xys['label']

        # this makes sure the order of features are normalized, and each
        # dimension of xys_2d is consistent with feature_names

        feature_names = sorted(xys.keys())
        feature_names.remove('label')
        feature_names.remove('content_id')

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
        if self.norm_type == 'whiten':
            self.mus = np.mean(xys_2d, axis=0)
            self.sds = np.std(xys_2d, axis=0)
        elif self.norm_type == 'rescale_0to1':
            self.fmins = np.min(xys_2d, axis=0)
            self.fmaxs = np.max(xys_2d, axis=0)
        elif self.norm_type == 'rescale_minus1to1':
            self.fmins = np.min(xys_2d, axis=0)
            self.fmaxs = np.max(xys_2d, axis=0)
        elif self.norm_type == 'none':
            pass
        else:
            assert False, 'Incorrect feature normalization type selected: {}'. \
                format(self.norm_type)

    def _normalize_xys(self, xys_2d):
        if self.norm_type == 'whiten':
            xys_2d -= self.mus
            xys_2d /= self.sds
        elif self.norm_type == 'rescale_0to1':
            xys_2d = 1.0 / (self.fmaxs - self.fmins) * (xys_2d - self.fmins)
        elif self.norm_type == 'rescale_minus1to1':
            xys_2d = 2.0 / (self.fmaxs - self.fmins) * (xys_2d - self.fmins) - 1
        elif self.norm_type == 'none':
            pass
        else:
            assert False, 'Incorrect feature normalization type selected: {}' \
                .format(self.norm_type)
        return xys_2d

    def denormalize_ys(self, ys_vec):
        if self.norm_type == 'whiten':
            ys_vec *= self.sds[0]
            ys_vec += self.mus[0]
        # elif self.norm_type == 'rescale_0to1':
        # for backward compatibility, use the following for older model files:
        elif self.norm_type == 'rescale_0to1' or self.norm_type == 'rescale1':
            ys_vec *= (self.fmaxs[0] - self.fmins[0])
            ys_vec += self.fmins[0]
        # elif self.norm_type == 'rescale_minus1to1':
        # for backward compatibility, use the following for older model files:
        elif self.norm_type == 'rescale_minus1to1' or self.norm_type == 'rescale2':
            ys_vec += 1
            ys_vec /= 2.0
            ys_vec *= (self.fmaxs[0] - self.fmins[0])
            ys_vec += self.fmins[0]
        elif self.norm_type == 'none':
            pass
        else:
            assert False, 'Incorrect feature normalization type selected: {}'. \
                format(self.norm_type)
        return ys_vec

    def normalize_xs(self, xs_2d):
        if self.norm_type == 'whiten':
            xs_2d -= self.mus[1:]
            xs_2d /= self.sds[1:]
        # elif self.norm_type == 'rescale_0to1':
        # for backward compatibility, use the following for older model files:
        elif self.norm_type == 'rescale_0to1' or self.norm_type == 'rescale1':
            xs_2d = 1.0 / (self.fmaxs[1:] - self.fmins[1:]) * \
                    (xs_2d - self.fmins[1:])
        # elif self.norm_type == 'rescale_minus1to1':
        # for backward compatibility, use the following for older model files:
        elif self.norm_type == 'rescale_minus1to1' or self.norm_type == 'rescale2':
            xs_2d = 2.0 / (self.fmaxs[1:] - self.fmins[1:]) * \
                    (xs_2d - self.fmins[1:]) - 1
        elif self.norm_type == 'none':
            pass
        else:
            assert False, 'Incorrect feature normalization type selected: {}' \
                .format(self.norm_type)
        return xs_2d

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

class NusvrTrainTestModel(TrainTestModel):

    TYPE = 'nusvr'

    @staticmethod
    def _train(model_param, xys_2d):
        """
        :param model_param:
        :param xys_2d:
        :return:
        """
        kernel = model_param.setdefault('kernel', 'rbf')
        degree = model_param.setdefault('degree', 3)
        gamma = model_param.setdefault('gamma', 0.0)
        coef0 = model_param.setdefault('coef0', 0.0)
        tol = model_param.setdefault('tol', 0.001)
        C = model_param.setdefault('C', 1.0)
        nu = model_param.setdefault('nu', 0.5)
        shrinking = model_param.setdefault('shrinking', True)
        cache_size= model_param.setdefault('cache_size', 200)
        verbose= model_param.setdefault('verbose', False)
        max_iter= model_param.setdefault('max_iter', -1)

        from sklearn.svm import NuSVR
        model = NuSVR(kernel=kernel,
                      degree=degree,
                      nu=nu,
                      gamma=gamma,
                      coef0=coef0,
                      tol=tol,
                      C=C,
                      shrinking=shrinking,
                      cache_size=cache_size,
                      verbose=verbose,
                      max_iter=max_iter
                      )
        model.fit(xys_2d[:, 1:], np.ravel(xys_2d[:, 0]))

        return model

class LibsvmnusvrTrainTestModel(TrainTestModel):

    TYPE = 'libsvmnusvr'

    sys.path.append(config.ROOT + "/libsvm/python")
    import svmutil

    # override
    def to_file(self, filename):
        assert self.model is not None, "Nothing to save..."

        self.svmutil.svm_save_model(filename + '.model', self.model)

        model_info = {}
        model_info['params'] = self.param_dict
        model_info['mu'] = self.mus
        model_info['sd'] = self.sds
        model_info['fmin'] = self.fmins
        model_info['fmax'] = self.fmaxs
        model_info['norm_type'] = self.norm_type
        model_info['feature_names'] = self.feature_names

        import joblib
        joblib.dump(model_info, filename, compress=9)

    # override
    @classmethod
    def from_file(cls, filename, logger):
        train_test_model = cls(param_dict={}, logger=logger)

        model = cls.svmutil.svm_load_model(filename + '.model')

        import joblib
        model_info = joblib.load(filename)

        train_test_model.param_dict = model_info['params']
        train_test_model.model = model
        train_test_model.mus = model_info['mu']
        train_test_model.sds = model_info['sd']
        train_test_model.fmins = model_info['fmin']
        train_test_model.fmaxs = model_info['fmax']

        # self.feature_names = model_info['feature_names']
        # for backward compatibility, use the following for older model files:
        train_test_model.feature_names = model_info['feature_names'] \
            if 'feature_names' in model_info else model_info['X_featurelabels']

        train_test_model.norm_type = model_info['norm_type']

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
        # m = svm_load_model(svm_file)
        # score, _, _ = svm_predict([0], X, m)
        # invoke the libsvm python script, for retraining purposes
        kernel = model_param.setdefault('kernel', 'rbf')
        # degree = model_param.setdefault('degree', 3)
        gamma = model_param.setdefault('gamma', 0.0)
        # coef0 = model_param.setdefault('coef0', 0.0)
        # tol = model_param.setdefault('tol', 0.001)
        C = model_param.setdefault('C', 1.0)
        nu = model_param.setdefault('nu', 0.5)
        # shrinking = model_param.setdefault('shrinking', True)
        cache_size= model_param.setdefault('cache_size', 200)
        # verbose= model_param.setdefault('verbose', False)
        # max_iter= model_param.setdefault('max_iter', -1)

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
                                        '-n',nu,
                                        '-m', cache_size])
        # convert to list! (silly step)
        f = list(xys_2d[:, 1:])
        for i, item in enumerate(f):
            f[i] = list(item)
        prob = cls.svmutil.svm_problem(xys_2d[:, 0], f)
        model = cls.svmutil.svm_train(prob, param)

        return model

class RandomForestTrainTestModel(TrainTestModel):

    TYPE = 'randomforest'

    @staticmethod
    def _train(model_param, xys_2d):
        """
        random forest regression
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        :param model_param:
        :param xys_2d:
        :return:
        """
        n_estimators = model_param.setdefault('n_estimators', 10)
        criterion = model_param.setdefault('criterion', 'mse')
        max_depth = model_param.setdefault('max_depth', None)
        min_samples_split = model_param.setdefault('min_samples_split', 2)
        min_samples_leaf = model_param.setdefault('min_samples_leaf', 1)
        min_weight_fraction_leaf = model_param.setdefault('min_weight_fraction_leaf', 0.0)
        max_features = model_param.setdefault('max_features', 'auto')
        max_leaf_nodes = model_param.setdefault('max_leaf_nodes', None)
        bootstrap = model_param.setdefault('bootstrap', True)
        oob_score = model_param.setdefault('oob_score', False)
        n_jobs = model_param.setdefault('n_jobs', 1)
        random_state = model_param.setdefault('random_state', None)
        verbose = model_param.setdefault('verbose', 0)
        #warm_start = model_param.setdefault('warm_start', False)

        from sklearn import ensemble
        model = ensemble.RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            # min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            # max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )
        model.fit(xys_2d[:, 1:], np.ravel(xys_2d[:, 0]))

        return model


class TrainTestModel2(object):

    def __init__(self, param_dict, logger=None):
        """
        :param param_dict: contains model parameters
        :param logger:
        :return:
        """
        self.param_dict = param_dict
        self.logger = logger

    def _assert_trained(self):

        assert hasattr(self, 'model_dict'), \
            "Must train first to generate model_dict."

        # usually get from Result._get_ordered_list_score_key() except for
        # customly constructed
        assert 'ordered_feature_names' in self.model_dict

        assert 'model' in self.model_dict

        assert 'norm_type' in self.model_dict
        norm_type = self.model_dict['norm_type']
        assert (   norm_type == 'none'
                or norm_type == 'normal'
                or norm_type == 'clipped')

        if norm_type == 'normal':
            assert 'mus' in self.model_dict
            assert 'sds' in self.model_dict

        if norm_type == 'clipped':
            assert 'fmins' in self.model_dict
            assert 'fmaxs' in self.model_dict

    def to_file(self, filename):

        self._assert_trained()

        info_to_save = {}
        info_to_save['param_dict'] = self.param_dict
        info_to_save['model_dict'] = self.model_dict

        import joblib
        joblib.dump(info_to_save, filename, compress=9)

    @property
    def ordered_feature_names(self):
        self._assert_trained()
        return self.model_dict['ordered_feature_names']

    @classmethod
    def from_file(cls, filename, logger):

        train_test_model = cls(param_dict={}, logger=logger)

        import joblib
        info_loaded = joblib.load(filename)

        train_test_model.param_dict = info_loaded['param_dict']
        train_test_model.model_dict = info_loaded['model_dict']

        return train_test_model

    # @staticmethod
    # def get_xs_from_results(results, indexs=None):
    #     """
    #     :param results: list of BasicResult
    #     :param indexs: indices of results to be used
    #     :return:
    #     """
    #     feature_names = results[0].get_ordered_list_score_key()
    #     xs = {}
    #     for name in feature_names:
    #         if indexs is None:
    #             _results = results
    #         else:
    #             _results = map(lambda i:results[i], indexs)
    #         xs[name] = np.array(map(lambda result: result[name], _results))
    #     return xs
    #
    # @staticmethod
    # def get_ys_from_results(results, indexs=None):
    #     """
    #     :param results: list of BasicResult
    #     :param indexs: indices of results to be used
    #     :return:
    #     """
    #     ys = {}
    #
    #     if indexs is None:
    #         _results = results
    #     else:
    #         _results = map(lambda i:results[i], indexs)
    #
    #     ys['label'] = \
    #         np.array(map(lambda result: result.asset.groundtruth, _results))
    #     ys['content_id'] = \
    #         np.array(map(lambda result: result.asset.content_id, _results))
    #
    #     return ys
    #
    # @classmethod
    # def get_xys_from_results(cls, results, indexs=None):
    #     """
    #     :param results: list of BasicResult
    #     :param indexs: indices of results to be used
    #     :return:
    #     """
    #     xys = {}
    #     xys.update(cls.get_xs_from_results(results, indexs))
    #     xys.update(cls.get_ys_from_results(results, indexs))
    #     return xys


    # ========================== begin of legacy ===============================

    # below is for the purpose of reading a legacy test text file, and to ensure
    # the other code in the class produces bit-exact results as before

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

    @staticmethod
    def _predict(model, xs_2d):

        ys_label_pred = model.predict(xs_2d)

        return ys_label_pred

    def train(self, xys):

        assert 'label' in xys

        ys_vec = xys['label']

        # TODO: modify
        # this makes sure the order of features are normalized, and each
        # dimension of xys_2d is consistent with feature_names

        feature_names = sorted(xys.keys())
        feature_names.remove('label')
        feature_names.remove('content_id')

        # TODO: continue

    def predict(self, xs):

        self._assert_trained()

        for name in self.ordered_feature_names:
            assert name in xs

        xs_2d = []
        for name in self.ordered_feature_names:
            if xs_2d == []:
                xs_2d = np.matrix(xs[name]).T
            else:
                xs_2d = np.hstack((xs_2d, np.matrix(xs[name]).T))
        xs_2d = np.array(xs_2d)

        # normalize xs
        xs_2d = self.normalize_xs(xs_2d)

        # predict
        model = self.model_dict['model']
        ys_label_pred = self._predict(model, xs_2d)

        # denormalize ys
        ys_label_pred = self.denormalize_ys(ys_label_pred)

        return ys_label_pred

    @staticmethod
    def get_stats(ys_label, ys_label_pred):

        import scipy.stats

        # MSE
        mse = np.mean(np.power(np.array(ys_label) - np.array(ys_label_pred), 2.0))

        # spearman
        srcc, _ = scipy.stats.spearmanr(ys_label, ys_label_pred)

        # pearson
        pcc, _ = scipy.stats.pearsonr(ys_label, ys_label_pred)

        # kendall
        kendall, _ = scipy.stats.kendalltau(ys_label, ys_label_pred)

        stats = {}
        stats['MSE'] = mse
        stats['SRCC'] = srcc
        stats['PCC'] = pcc
        stats['KENDALL'] = kendall

        # append raw
        stats['ys_label'] = list(ys_label)
        stats['ys_label_pred'] = list(ys_label_pred)

        return stats

    @classmethod
    def aggregate_stats_list(cls, stats_list):
        aggregate_ys_label = []
        aggregate_ys_label_pred = []
        for stats in stats_list:
            aggregate_ys_label += stats['ys_label']
            aggregate_ys_label_pred += stats['ys_label_pred']
        return cls.get_stats(aggregate_ys_label, aggregate_ys_label_pred)

    @staticmethod
    def get_objective_score(stats, type='SRCC'):
        """
        Objective score is something to MAXIMIZE. e.g. SRCC, or -MSE.
        :param stats:
        :param type:
        :return:
        """
        if type == 'SRCC':
            return stats['SRCC']
        elif type == 'PCC':
            return stats['PCC']
        elif type == 'KENDALL':
            return stats['KENDALL']
        elif type == 'MSE':
            return -stats['MSE']
        else:
            assert False, 'Unknow type: {} for get_objective_score().'.format(type)

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

    def evaluate(self, xs, ys):
        ys_label_pred = self.predict(xs)
        ys_label = ys['label']

        return self.get_stats(ys_label, ys_label_pred)


class LibsvmnusvrTrainTestModel2(TrainTestModel):

    @classmethod
    def from_raw_file(cls, filename, additional_model_dict):
        """
        Construct from raw libsvm model file.
        :param filename:
        :param additional_model_dict:
        :return:
        """
        pass

    pass

class RandomForestTrainTestModel2(TrainTestModel):
    pass