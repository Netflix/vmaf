__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import numpy as np


class ListStats(object):
    """
    >>> test_list = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    >>> "%0.4f" % ListStats.total_variation(test_list)
    '1.5556'
    >>> np.mean(test_list)
    8.0
    >>> np.median(test_list)
    8.0
    >>> ListStats.lp_norm(test_list, 1.0)
    8.0
    >>> "%0.4f" % ListStats.lp_norm(test_list, 3.0)
    '10.5072'
    >>> "%0.2f" % ListStats.perc1(test_list)
    '1.09'
    >>> "%0.2f" % ListStats.perc5(test_list)
    '1.45'
    >>> "%0.2f" % ListStats.perc10(test_list)
    '1.90'
    >>> "%0.2f" % ListStats.perc20(test_list)
    '2.80'
    >>> ListStats.nonemean([None, None, 1, 2])
    1.5
    >>> ListStats.nonemean([3, 4, 1, 2])
    2.5
    >>> ListStats.nonemean([None, None, None])
    nan
    >>> "%0.4f" % ListStats.harmonic_mean(test_list)
    '4.5223'
    >>> "%0.4f" % ListStats.lp_norm(test_list, 2.0)
    '9.5394'
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
    def perc1(my_list):
        return np.percentile(my_list, 1)

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
        print("Min: {min}, Max: {max}, Median: {median}, Mean: {mean}," \
              " Variance: {var}, Total_variation: {total_var}".format(
            min=np.min(my_list), max=np.max(my_list),
            median=np.median(my_list), mean=np.mean(my_list),
            var=np.var(my_list),
            total_var=ListStats.total_variation(my_list)))

    @staticmethod
    def print_moving_average_stats(my_list, n, type='exponential', decay=-1):
        moving_avg_list = ListStats.moving_average(my_list, n, type, decay)
        ListStats.print_stats(moving_avg_list)

    @staticmethod
    def nonemean(my_list):
        return np.mean(list(filter(lambda x: x is not None, my_list)))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
