__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import numpy as np

class StatsList(object):
    """
    >>> test_dict = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    >>> StatsList.min(test_dict)
    1
    >>> StatsList.max(test_dict)
    15
    >>> StatsList.median(test_dict)
    8.0
    >>> StatsList.mean(test_dict)
    8.0
    >>> StatsList.stddev(test_dict)
    5.196152422706632
    >>> StatsList.var(test_dict)
    27.0
    >>> StatsList.percentile(test_dict, 50)
    8.0
    >>> StatsList.percentile(test_dict, 80)
    13.200000000000001
    >>> StatsList.moving_average(test_dict, 2)
    array([  2.26894142,   2.26894142,   2.26894142,   3.26894142,
             4.26894142,   6.61364853,  11.26894142,  12.26894142,
            13.26894142,  14.26894142])
    >>> StatsList.moving_average(test_dict, 5)
    array([  4.08330969,   4.08330969,   4.08330969,   4.08330969,
             4.08330969,   4.08330969,   5.81552983,   7.7557191 ,
             9.96294602,  12.51305607])
    """

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
        return np.percentile(list, q)

    @staticmethod
    def total_var(list):
        abs_diff_scores = np.absolute(np.diff(list))
        return np.mean(abs_diff_scores)

    @staticmethod
    def moving_average(list, n, type='exponential', decay=-1):
        """
        compute an n period moving average.
        :param list:
        :param n:
        :param type: 'simple' | 'exponential'
        :param decay:
        :return:
        """
        x = np.asarray(list)
        if type == 'simple':
            weights = np.ones(n)
        elif type == 'exponential':
            weights = np.exp(np.linspace(decay, 0., n))
        else:
            assert False, "Unknown type: {}.".format(type)

        weights /= weights.sum()

        a = np.convolve(x, weights, mode='full')[:len(x)]
        a[:n] = a[n]
        return a

if __name__ == '__main__':
    import doctest
    doctest.testmod()
