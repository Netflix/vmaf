import numpy as np

from vmaf.tools.interpolation_utils import InterpolationUtils

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class BDrateCalculator(object):
    """
    BD rate calculator. Implementation validated against JCTVC-E137:
    http://phenix.it-sudparis.eu/jct/doc_end_user/documents/5_Geneva/wg11/JCTVC-E137-v1.zip
    """

    REJECTED_BD_RATE_NO_OVERLAP = -10000
    REJECTED_BD_RATE_NON_MONOTONIC = -20000
    REJECTED_BD_RATE_MESSED_UP_RATE = -30000
    REJECTED_BALANCE_BAD_MEASUREMENTS = -40000

    @staticmethod
    def _dedup_and_order(set_):
        return sorted(list(set(set_)), key=lambda x: x[0])

    @classmethod
    def CalcBDRate(cls, setA, setB):

        # ==== added by zli =======
        setA = cls._dedup_and_order(setA)
        setB = cls._dedup_and_order(setB)
        # ==== added by zli =======

        assert not (len(setA) < 4 or len(setB) < 4), \
            "Problem with input RD point lists. setA is size {}, " \
            "setB is size {}".format(len(setA), len(setB))

        if not cls.isCurveMonotonic(setA):
            raise AssertionError(cls.REJECTED_BD_RATE_NON_MONOTONIC)

        if not cls.isCurveMonotonic(setB):
            raise AssertionError(cls.REJECTED_BD_RATE_NON_MONOTONIC)

        if not cls.ratesLookOkay(setA):
            raise AssertionError(cls.REJECTED_BD_RATE_MESSED_UP_RATE)

        if not cls.ratesLookOkay(setB):
            raise AssertionError(cls.REJECTED_BD_RATE_MESSED_UP_RATE)

        minMainPSNR = setA[0][1]
        maxMainPSNR = setA[-1][1]
        minHighPSNR = setB[0][1]
        maxHighPSNR = setB[-1][1]

        minPSNR = max(minMainPSNR, minHighPSNR)
        maxPSNR = min(maxMainPSNR, maxHighPSNR)

        # no overlap, so mark it in a special way
        if minPSNR >= maxPSNR:
            return cls.REJECTED_BD_RATE_NO_OVERLAP

        vA = cls.bdrint(setA, minPSNR, maxPSNR)
        vB = cls.bdrint(setB, minPSNR, maxPSNR)

        avg = (vB - vA) / (maxPSNR - minPSNR)
        return np.power(10, avg) - 1

    @staticmethod
    def isCurveMonotonic(set_):
        for i in range(len(set_) - 1):

            # ==== added by zli =======
            if set_[i][0] >= set_[i + 1][0]:
                return False
            # ==== added by zli =======

            # ==== added by zli =======
            # if set_[i][1] > set_[i + 1][1]:
            if set_[i][1] >= set_[i + 1][1]:
            # ==== added by zli =======
                return False

        return True

    @staticmethod
    def ratesLookOkay(set_):
        for i in range(len(set_)):
            if set_[i][0] == 0:
                return False
        return True

    # // BD-rate calculation for arbitrary number (N) points
    # // cf. https://www.mathworks.com/moler/interp.pdf, sections 3.3 - 3.4
    @staticmethod
    def bdrint(rdPointsList, minPSNR, maxPSNR):

        N = len(rdPointsList)

        log_rate = []
        log_dist = []
        H = []
        delta = []
        d = []
        c = []
        b = []

        InterpolationUtils.computeParamsForSegments(rdPointsList, log_rate, log_dist, H, delta, d, c, b, True)

        # // cubic function is rate(i) + s*(d(i) + s*(c(i) + s*(b(i))) where s = x - dist(i)
        # // or rate(i) + s*d(i) + s*s*c(i) + s*s*s*b(i)
        # // primitive is s*rate(i) + s*s*d(i)/2 + s*s*s*c(i)/3 + s*s*s*s*b(i)/4

        result = 0.0

        for i in range(N - 1):
            s0 = log_dist[i]
            s1 = log_dist[i + 1]

            # // clip s0 to valid range
            s0 = max(s0, minPSNR)
            s0 = min(s0, maxPSNR)
            s0 -= log_dist[i]

            # // clip s1 to valid range
            s1 = max(s1, minPSNR)
            s1 = min(s1, maxPSNR)
            s1 -= log_dist[i]

            if s1 > s0:
                result += (s1 - s0) * log_rate[i]
                result += (s1 * s1 - s0 * s0) * d[i] / 2.0
                result += (s1 * s1 * s1 - s0 * s0 * s0) * c[i] / 3.0
                result += (s1 * s1 * s1 * s1 - s0 * s0 * s0 * s0) * b[i] / 4.0

        return result
