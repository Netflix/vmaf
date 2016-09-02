import numpy as np
import sklearn.metrics
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

from core.quality_runner import VmafQualityRunner
from core.result import Result
from tools.misc import get_file_name_without_extension
from tools.reader import YuvReader

# Copyright (c) 2016, Marco Tulio Correia Ribeiro
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

    @staticmethod
    def assert_explanations(exps, assets=None, ys=None, ys_pred=None):
        N = exps['feature_weights'].shape[0]
        assert N == exps['features_normalized'].shape[0]
        if assets is not None:
            assert N == len(assets)
        if ys is not None:
            assert N == len(ys['label'])
        if ys_pred is not None:
            assert N == len(ys_pred)
        return N

    @classmethod
    def print_explanations(cls, exps, assets=None, ys=None, ys_pred=None):

        # asserts
        N = cls.assert_explanations(exps, assets, ys, ys_pred)

        print "Features: {}".format(exps['feature_names'])

        for n in range(N):
            weights = exps['feature_weights'][n]
            features = exps['features_normalized'][n]

            asset = assets[n] if assets is not None else None
            y = ys['label'][n] if ys is not None else None
            y_pred = ys_pred[n] if ys_pred is not None else None

            print "{ref}".format(
                ref=get_file_name_without_extension(asset.ref_path) if
                asset is not None else "Asset {}".format(n))
            if asset is not None:
                print "\tDistorted: {dis}".format(
                    dis=get_file_name_without_extension(asset.dis_path))
            if y is not None:
                print "\tground truth: {y:.3f}".format(y=y)
            if y_pred is not None:
                print "\tpredicted: {y_pred:.3f}".format(y_pred=y_pred)
            print "\tfeature value: {}".format(features)
            print "\tfeature weight: {}".format(weights)

    @classmethod
    def plot_explanations(cls, exps, assets=None, ys=None, ys_pred=None):

        # asserts
        N = cls.assert_explanations(exps, assets, ys, ys_pred)

        figs = []
        for n in range(N):
            weights = exps['feature_weights'][n]
            features = exps['features_normalized'][n]

            asset = assets[n] if assets is not None else None
            y = ys['label'][n] if ys is not None else None
            y_pred = ys_pred[n] if ys_pred is not None else None

            img = None
            if asset is not None:
                w, h = asset.dis_width_height
                with YuvReader(filepath=asset.dis_path, width=w, height=h,
                               yuv_type=asset.yuv_type) as yuv_reader:
                    for yuv in yuv_reader:
                        img, _, _ = yuv
                        break
                assert img is not None

            title = ""
            if asset is not None:
                title += "{}\n".format(get_file_name_without_extension(asset.ref_path))
            if y is not None:
                title += "ground truth: {:.3f}\n".format(y)
            if y_pred is not None:
                title += "predicted: {:.3f}\n".format(y_pred)
            if title != "" and title[-1] == '\n':
                title = title[:-1]

            assert len(weights) == len(features)
            M = len(weights)

            fig = plt.figure()

            ax_top = plt.subplot(2, 1, 1)
            ax_left = plt.subplot(2, 2, 3)
            ax_right = plt.subplot(2, 2, 4, sharey=ax_left)

            if img is not None:
                ax_top.imshow(img, cmap='Greys_r')
            ax_top.get_xaxis().set_visible(False)
            ax_top.get_yaxis().set_visible(False)
            ax_top.set_title(title)

            pos = np.arange(M) + 0.1
            ax_left.barh(pos, features, color='b', label='feature')
            ax_left.set_xticks(np.arange(0, 1.1, 0.2))
            ax_left.set_yticks(pos + 0.35)
            ax_left.set_yticklabels(exps['feature_names'])
            ax_left.set_title('feature')

            ax_right.barh(pos, weights, color='r', label='weight')
            ax_right.get_yaxis().set_visible(False)
            ax_right.set_title('weight')

            plt.tight_layout()

            figs.append(fig)

        return figs

    @classmethod
    def select_from_exps(cls, exps, indexs):
        # asserts
        N = cls.assert_explanations(exps)
        for index in indexs:
            assert index < N
        exps2 = {
            'feature_weights': exps['feature_weights'][indexs, :],
            'features': exps['features'][indexs, :],
            'features_normalized': exps['features_normalized'][indexs, :],
            'feature_names': exps['feature_names']
        }
        return exps2


class VmafQualityRunnerWithLocalExplainer(VmafQualityRunner):
    """Same as VmafQualityRunner, except it outputs additional LocalExplainer
    results."""

    @classmethod
    def get_explanations_key(cls):
        return cls.get_scores_key() + '_exps'

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
        result_dict[self.get_explanations_key()] = exps # add local explanations
        return Result(asset, self.executor_id, result_dict)

    @classmethod
    def show_local_explanations(cls, results, indexs=None):
        """Plot local explanations of results

        :param results:
        :param indexs: a list of frame indices, or None. If None, will take the
        second frame.
        :return: figures of local explanation plots
        """

        # assert results are indeed generated by class
        for result in results:
            assert cls.get_explanations_key() in result.result_dict

        N = len(results)

        if indexs is None:
            indexs = [1] # default: second frame

        figss = []
        for n in range(N):

            exps = results[n][cls.get_explanations_key()]
            asset = results[n].asset
            exps2 = LocalExplainer.select_from_exps(exps, indexs)

            ys_pred = results[n][cls.get_scores_key()][indexs]

            N2 = LocalExplainer.assert_explanations(exps2)
            assets2 = [asset for _ in range(N2)]

            # LocalExplainer.print_explanations(exps2, assets=assets2, ys=None, ys_pred=ys_pred)
            figs = LocalExplainer.plot_explanations(exps2, assets=assets2, ys=None, ys_pred=ys_pred)
            figss.append(figs)

        return figss