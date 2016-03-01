__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import numpy as np

class StatsList(object):

    @staticmethod
    def min(dict):
        return np.min(dict)

    @staticmethod
    def max(dict):
        return np.max(dict)

    @staticmethod
    def median(dict):
        return np.median(dict)

    @staticmethod
    def mean(dict):
        return np.mean(dict)

    @staticmethod
    def stddev(dict):
        return np.std(dict)

    @staticmethod
    def var(dict):
        return np.var(dict)

    @staticmethod
    def percentile(dict, q):
        return np.percentile(dict,q)

    @staticmethod
    def total_var(dict):
        abs_diff_scores = np.absolute(np.diff(dict))
        return np.mean(abs_diff_scores)

    @staticmethod
    def moving_average(dict, n, type='exponential'):
        """
        compute an n period moving average.

        type is 'simple' | 'exponential'

        """
        x = np.asarray(dict)
        if type == 'simple':
            weights = np.ones(n)
        else:
            weights = np.exp(np.linspace(-1., 0., n))

        weights /= weights.sum()

        a = np.convolve(x, weights, mode='full')[:len(x)]
        a[:n] = a[n]
        return a