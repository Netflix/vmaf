__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import numpy as np

class StatsList(object):

    @staticmethod
    def min(list):
        return np.min(list)

    @staticmethod
    def max(list):
        return np.max(list)

    @staticmethod
    def median(list):
        return np.median(list)

    @staticmethod
    def mean(list):
        return np.mean(list)

    @staticmethod
    def stddev(list):
        return np.std(list)

    @staticmethod
    def var(list):
        return np.var(list)

    @staticmethod
    def percentile(list, q):
        return np.percentile(list,q)

    @staticmethod
    def total_var(list):
        abs_diff_scores = np.absolute(np.diff(list))
        return np.mean(abs_diff_scores)

    @staticmethod
    def moving_average(list, n, type='exponential', decay=-1):
        """
        compute an n period moving average.

        type is 'simple' | 'exponential'

        """
        x = np.asarray(list)
        if type == 'simple':
            weights = np.ones(n)
        else:
            weights = np.exp(np.linspace(decay, 0., n))

        weights /= weights.sum()

        a = np.convolve(x, weights, mode='full')[:len(x)]
        a[:n] = a[n]
        return a