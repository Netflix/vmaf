from vmaf.tools.decorator import override

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import numpy as np
import scipy.linalg

from vmaf.core.train_test_model import TrainTestModel, RegressorMixin


class NiqeTrainTestModel(TrainTestModel, RegressorMixin):

    TYPE = 'NIQE'
    VERSION = "0.1"

    @classmethod
    @override(TrainTestModel)
    def _assert_dimension(cls, feature_names, results):
        # Allow input to be list
        # For each result, the dimension of each result[feature_name]
        # should be consistent
        assert isinstance(results[0][feature_names[0]], list)
        for result in results:
            len0 = len(result[feature_names[0]])
            for name in feature_names[1:]:
                assert len(result[name]) == len0

    @classmethod
    @override(TrainTestModel)
    def get_xs_from_results(cls, results, indexs=None, aggregate=False, features=None):
        """
        override by altering aggregate
        default to False
        """
        return super(NiqeTrainTestModel, cls).get_xs_from_results(
            results, indexs, aggregate, features)

    @classmethod
    @override(TrainTestModel)
    def get_xys_from_results(cls, results, indexs=None, aggregate=False):
        """
        override by altering aggregate
        default to False
        """
        return super(NiqeTrainTestModel, cls).get_xys_from_results(
            results, indexs, aggregate)

    @override(TrainTestModel)
    def train(self, xys):
        self.model_type = self.TYPE

        assert 'label' in xys
        ys_vec = xys['label'] # for NIQE, ys never used for training

        # this makes sure the order of features are normalized, and each
        # dimension of xys_2d is consistent with feature_names
        feature_names = sorted(xys.keys())
        feature_names.remove('label')
        feature_names.remove('content_id')
        self.feature_names = feature_names

        num_samples = len(xys[feature_names[0]])

        xs_2d = []
        for i_sample in range(num_samples):
            xs_2d_ = np.vstack(map(
                lambda feature_name: xys[feature_name][i_sample], feature_names)
            ).T
            xs_2d.append(xs_2d_)
        xs_2d = np.vstack(xs_2d)

        # no normalization for NIQE
        self.norm_type = 'none'

        # compute NIQE
        mu = np.mean(xs_2d, axis=0)
        cov = np.cov(xs_2d.T)

        self.model = {'mu': mu, 'cov': cov}

    @override(TrainTestModel)
    def predict(self, xs):
        self._assert_trained()

        for name in self.feature_names:
            assert name in xs

        num_samples = len(xs[self.feature_names[0]])

        # predict per sample
        ys_label_pred = []
        for i_sample in range(num_samples):
            xs_2d_ = np.vstack(list(map(
                lambda feature_name: xs[feature_name][i_sample],
                self.feature_names))
            ).T

            # no normalization for NIQE

            if xs_2d_.shape[0] < 2:
                ys_label_pred_ = None # NIQE won't work for single patch
            else:
                ys_label_pred_ = self._predict(self.model, xs_2d_)

            ys_label_pred.append(ys_label_pred_)

        return {'ys_label_pred': ys_label_pred}

    @classmethod
    def _predict(cls, model, xs_2d):
        pop_mu = model['mu'].astype(float)
        pop_cov = model['cov'].astype(float)
        feats = xs_2d
        sample_mu = np.mean(feats, axis=0)
        sample_cov = np.cov(feats.T)
        X = sample_mu - pop_mu
        covmat = ((pop_cov+sample_cov)/2.0)
        pinvmat = scipy.linalg.pinv(covmat)
        d1 = np.sqrt(np.dot(np.dot(X, pinvmat), X))
        return d1