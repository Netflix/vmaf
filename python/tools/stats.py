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
    >>> StatsList.variance(test_dict)
    27.0
    >>> StatsList.percentile(test_dict, 50)
    8.0
    >>> StatsList.percentile(test_dict, 80)
    13.200000000000001
    >>> StatsList.total_variation(test_dict)
    1.5555555555555556
    >>> StatsList.moving_average(test_dict, 2)
    array([  2.26894142,   2.26894142,   2.26894142,   3.26894142,
             4.26894142,   6.61364853,  11.26894142,  12.26894142,
            13.26894142,  14.26894142])
    >>> StatsList.moving_average(test_dict, 5)
    array([  4.08330969,   4.08330969,   4.08330969,   4.08330969,
             4.08330969,   4.08330969,   5.81552983,   7.7557191 ,
             9.96294602,  12.51305607])

    >>> StatsList.print_stats(test_dict)
    Min: 1, Max: 15, Median: 8.0, Mean: 8.0, Variance: 27.0, Total_variation: 1.55555555556

    >>> StatsList.print_moving_average_stats(test_dict, 3)
    Min: 2.67984333217, Max: 13.6798433322, Median: 4.64565264023, Mean: 6.61976499826, Variance: 18.625918874, Total_variation: 1.22222222222
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
    def variance(list):
        return np.var(list)

    @staticmethod
    def percentile(list, q):
        return np.percentile(list, q)

    @staticmethod
    def total_variation(list):
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

    @staticmethod
    def print_stats(list):
        print "Min: {min}, Max: {max}, Median: {median}, Mean: {mean}," \
              " Variance: {var}, Total_variation: {total_var}".format(
            min=StatsList.min(list), max=StatsList.max(list),
            median=StatsList.median(list), mean=StatsList.mean(list),
            var=StatsList.variance(list),
            total_var=StatsList.total_variation(list))

    @staticmethod
    def print_moving_average_stats(list, n, type='exponential', decay=-1):
        moving_avg_list = StatsList.moving_average(list, n, type, decay)
        StatsList.print_stats(moving_avg_list)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
