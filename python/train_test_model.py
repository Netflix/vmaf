__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import numpy as np
from tools import indices

class TrainTestModel(object):

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
            "Must train first to generate model_dict before save to file."

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

        train_test_model = cls(param_dict={},
                               logger=logger)

        import joblib
        info_loaded = joblib.load(filename)

        train_test_model.param_dict = info_loaded['param_dict']
        train_test_model.model_dict = info_loaded['model_dict']

        return train_test_model

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


class LibsvmnusvrTrainTestModel(TrainTestModel):

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

class RandomForestTrainTestModel(TrainTestModel):
    pass