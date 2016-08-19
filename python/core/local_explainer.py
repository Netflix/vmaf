import numpy as np
import sklearn.metrics
from sklearn.linear_model import Ridge

from core.quality_runner import VmafQualityRunner
from core.result import Result

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class LocalExplainer(object):
    """Explains a TrainTestModel on a local data point.
    Adapted from:
    Lime: Explaining the predictions of any machine learning classifier
    https://github.com/marcotcr/lime"""

    def __init__(self,
                 neighbor_std=1.0,
                 neighbor_samples=5000,
                 distance_metric='euclidean',
                 kernel_width=3,
                 model_regressor=None,
                 ):
        """Init function.

        :param neighbor_std: standard deviation of neighborhood sampled
        :param neighbor_samples: number of samples of neighborhood
        :param distance_metric: distance metric used
        :param kernel_width: width for kernel function
        :param model_regressor: regressor to train local linear model. If None, use Ridge
        """
        self.neighbor_std = neighbor_std
        self.neighbor_samples = neighbor_samples
        self.distance_metric = distance_metric
        self.kernel_fn = lambda d: np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
        self.model_regressor = Ridge(alpha=1, fit_intercept=True) \
                               if model_regressor is None else model_regressor

    def _assert_model(self, train_test_model):

        train_test_model._assert_trained()

        assert hasattr(train_test_model, '_to_tabular_xs'), \
            'train_test_model must have a method _to_tabular_xs().'

        assert train_test_model.norm_type != 'none', \
            'If train_test_model has not been normalized, ' \
            'the sampled neighborhood may not be of the right shape.'

    def explain(self, train_test_model, xs):
        """Explain data points.

        :param train_test_model: a trained TrainTestModel object
        :param xs: same xs as in TrainTestModel.predict(xs)
        :return: exps: explanations, where exps['feature_weights'] has the
        feature weights (in num_sample x num_feature 2D array)
        """

        self._assert_model(train_test_model)

        feature_names = train_test_model.feature_names
        for name in feature_names:
            assert name in xs

        xs_2d = train_test_model._to_tabular_xs(feature_names, xs)

        xs_2d_unnormalized = xs_2d.copy()

        # normalize xs
        xs_2d = train_test_model.normalize_xs(xs_2d)

        # for each row of xs, repreat feature of a unit (e.g. frame),
        # generate a new 2d_array by sampling its neighborhood
        n_sample, n_feature = xs_2d.shape
        feature_weights = np.zeros([n_sample, n_feature])
        for i_sample in range(n_sample):

            # generate neighborhood samples
            x_row = xs_2d[i_sample, :]
            xs_2d_neighbor = np.random.randn(self.neighbor_samples, n_feature) \
                             * self.neighbor_std
            xs_2d_neighbor += np.tile(x_row, (self.neighbor_samples, 1))

            # add center to first row
            xs_2d_neighbor = np.vstack([x_row, xs_2d_neighbor])

            # calculate distance to center
            distances = sklearn.metrics.pairwise_distances(
                xs_2d_neighbor,
                xs_2d_neighbor[0].reshape(1, -1),
                metric=self.distance_metric
            ).ravel()
            sample_weight = self.kernel_fn(distances)

            # predict
            ys_label_pred_neighbor = train_test_model._predict(
                                     train_test_model.model, xs_2d_neighbor)

            # take xs_2d_neighbor and ys_label_pred_neighbor, train a linear
            # model
            self.model_regressor.fit(xs_2d_neighbor, ys_label_pred_neighbor,
                                     sample_weight=sample_weight)
            feature_weight = self.model_regressor.coef_.copy()
            feature_weights[i_sample, :] = feature_weight

        exps = {
            'feature_weights': feature_weights,
            'features': xs_2d_unnormalized,
            'features_normalized': xs_2d,
            'feature_names': feature_names
        }

        return exps


class VmafQualityRunnerWithLocalExplainer(VmafQualityRunner):
    """Same as VmafQualityRunner, except it outputs additional LocalExplainer
    results."""

    def _run_on_asset(self, asset):
        # Override VmafQualityRunner._run_on_asset(self, asset), by adding
        # additional local explanation info.
        vmaf_fassembler = self._get_vmaf_feature_assembler_instance(asset)
        vmaf_fassembler.run()
        feature_result = vmaf_fassembler.results[0]
        model = self._load_model()
        xs = model.get_per_unit_xs_from_a_result(feature_result)
        ys_pred = model.predict(xs)
        ys_pred = self.clip_score(model, ys_pred)

        if self.optional_dict2 is not None and \
           'explainer' in self.optional_dict2:
            explainer = self.optional_dict2['explainer']
        else:
            explainer = LocalExplainer()

        exps = explainer.explain(model, xs)
        result_dict = {}
        result_dict.update(feature_result.result_dict) # add feature result
        result_dict[self.get_scores_key()] = ys_pred # add quality score
        result_dict[self.get_scores_key() + '_exps'] = exps # add local explanations
        return Result(asset, self.executor_id, result_dict)