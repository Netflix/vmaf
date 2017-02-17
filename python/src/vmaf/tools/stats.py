__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import numpy as np

class ListStats(object):
    """
    >>> test_list = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    >>> ListStats.total_variation(test_list)
    1.5555555555555556
    >>> ListStats.moving_average(test_list, 2)
    array([  2.26894142,   2.26894142,   2.26894142,   3.26894142,
             4.26894142,   6.61364853,  11.26894142,  12.26894142,
            13.26894142,  14.26894142])
    >>> ListStats.moving_average(test_list, 5)
    array([  4.08330969,   4.08330969,   4.08330969,   4.08330969,
             4.08330969,   4.08330969,   5.81552983,   7.7557191 ,
             9.96294602,  12.51305607])
    >>> ListStats.harmonic_mean(test_list)
    4.5222635212015483
    >>> np.mean(test_list)
    8.0
    >>> np.median(test_list)
    8.0
    >>> ListStats.lp_norm(test_list, 2.0)
    9.5393920141694561
    >>> ListStats.lp_norm(test_list, 1.0)
    8.0
    >>> ListStats.lp_norm(test_list, 3.0)
    10.507175744985801
    >>> ListStats.perc5(test_list)
    1.4500000000000002
    >>> ListStats.perc10(test_list)
    1.8999999999999999
    >>> ListStats.perc20(test_list)
    2.8000000000000003

    >>> ListStats.print_stats(test_list)
    Min: 1, Max: 15, Median: 8.0, Mean: 8.0, Variance: 27.0, Total_variation: 1.55555555556

    >>> ListStats.print_moving_average_stats(test_list, 3)
    Min: 2.67984333217, Max: 13.6798433322, Median: 4.64565264023, Mean: 6.61976499826, Variance: 18.625918874, Total_variation: 1.22222222222
    """

    @staticmethod
    def total_variation(my_list):
        abs_diff_scores = np.absolute(np.diff(my_list))
        return np.mean(abs_diff_scores)

    @staticmethod
    def moving_average(my_list, n, type='exponential', decay=-1):
        """
        compute an n period moving average.
        :param my_list:
        :param n:
        :param type: 'simple' | 'exponential'
        :param decay:
        :return:
        """
        x = np.asarray(my_list)
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
    def harmonic_mean(my_list):
        return 1.0 / np.mean(1.0 / (np.array(my_list) + 1.0)) - 1.0

    @staticmethod
    def lp_norm(my_list, p):
        return np.power(np.mean(np.power(np.array(my_list), p)), 1.0 / p)

    @staticmethod
    def perc5(my_list):
        return np.percentile(my_list, 5)

    @staticmethod
    def perc10(my_list):
        return np.percentile(my_list, 10)

    @staticmethod
    def perc20(my_list):
        return np.percentile(my_list, 20)

    @staticmethod
    def print_stats(my_list):
        print "Min: {min}, Max: {max}, Median: {median}, Mean: {mean}," \
              " Variance: {var}, Total_variation: {total_var}".format(
            min=np.min(my_list), max=np.max(my_list),
            median=np.median(my_list), mean=np.mean(my_list),
            var=np.var(my_list),
            total_var=ListStats.total_variation(my_list))

    @staticmethod
    def print_moving_average_stats(my_list, n, type='exponential', decay=-1):
        moving_avg_list = ListStats.moving_average(my_list, n, type, decay)
        ListStats.print_stats(moving_avg_list)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
