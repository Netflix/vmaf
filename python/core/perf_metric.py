import numpy as np
from numpy.linalg import lstsq
import scipy.stats

from core.mixin import TypeVersionEnabled

__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

class PerfMetric(TypeVersionEnabled):

    def __init__(self, groundtruths, predictions):
        """
        Performance metric on quality metrics
        :param groundtruths: either list of real numbers (aggregate scores like
        MOS or DMOS or MLE), or list of list of real numbers (list of raw scores)
        :param predictions: list of real numbers
        :return:
        """
        TypeVersionEnabled.__init__(self)
        self.groundtruths = groundtruths
        self.predictions = predictions
        self._assert_args()

    def _assert_args(self):
        assert len(self.groundtruths) == len(self.predictions)

    def evaluate(self, **kwargs):
        """
        :return: ret - a dictionary with 'score' and other keys
        """
        groundtruths, predictions = self._preprocess(self.groundtruths, self.predictions, **kwargs)
        result = self._evaluate(groundtruths, predictions, **kwargs)
        assert 'score' in result
        return result

class RawScorePerfMetric(PerfMetric):
    """
    Groundtruth is a list of raw scores (list of list of real numbers)
    """

    pass

class AggrScorePerfMetric(PerfMetric):
    """
    Groundtruth is a list of aggregate scores (list of real numbers)
    """

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
    def _preprocess(cls, groundtruths, predictions, **kwargs):
        aggre_method = kwargs['aggr_method'] if 'aggr_method' in kwargs else np.mean
        enable_mapping = kwargs['enable_mapping'] if 'enable_mapping' in kwargs else False

        groundtruths_ = map(
            lambda x: aggre_method(x) if hasattr(x, '__len__') else x,
            groundtruths)

        if enable_mapping:
            predictions_ = cls.sigmoid_adjust(predictions, groundtruths_)
        else:
            predictions_ = predictions

        return groundtruths_, predictions_

class RmsePerfMetric(AggrScorePerfMetric):

    TYPE = "RMSE"
    VERSION = "1.0"

    @staticmethod
    def _evaluate(groundtruths, predictions, **kwargs):
        rmse = np.sqrt(np.mean(np.power(np.array(groundtruths) - np.array(predictions), 2.0)))
        result = {'score': rmse}
        return result

class SrccPerfMetric(AggrScorePerfMetric):

    TYPE = "SRCC"
    VERSION = "1.0"

    @staticmethod
    def _evaluate(groundtruths, predictions, **kwargs):
        # spearman
        srcc, _ = scipy.stats.spearmanr(groundtruths, predictions)
        result = {'score': srcc}
        return result

class PccPerfMetric(AggrScorePerfMetric):

    TYPE = "PCC"
    VERSION = "1.0"

    @staticmethod
    def _evaluate(groundtruths, predictions, **kwargs):
        # pearson
        pcc, _ = scipy.stats.pearsonr(groundtruths, predictions)
        result = {'score': pcc}
        return result

class KendallPerfMetric(AggrScorePerfMetric):

    TYPE = "KENDALL"
    VERSION = "1.0"

    @staticmethod
    def _evaluate(groundtruths, predictions, **kwargs):
        # kendall
        kendall, _ = scipy.stats.kendalltau(groundtruths, predictions)
        result = {'score': kendall}
        return result
