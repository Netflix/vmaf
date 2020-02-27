import numpy as np

__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"


class InterpolationUtils(object):

    # /**
    #  * Compute PCHIP params and compute the rate from distortion (or PSNR).
    #  *
    #  * @param rdPointsList        List of RD points, minimum 4 needed
    #  * @param arrayDistInLogScale List of distortion values for which rate values are desired
    #  * @return List of rate values, either in log scale or linear scale
    #  */
    @classmethod
    def interpolateRateFromMetric(cls, rdPointsList, arrayDistInLogScale):

        N = len(rdPointsList)

        log_rate = []
        log_dist = []
        H = []
        delta = []
        d = []
        c = []
        b = []

        cls.computeParamsForSegments(rdPointsList, log_rate, log_dist, H, delta, d, c, b, False)

        arrayRate = []

        for distInLogScale in arrayDistInLogScale:
            distBegin = log_dist[0]
            yk = log_rate[0]
            segmentOfInterest = 0
            for i in range(N-1):
                if log_dist[i] <= distInLogScale and log_dist[i+1] >= distInLogScale:
                    distBegin = log_dist[i]
                    yk = log_rate[i]
                    segmentOfInterest = i
                    break

            rate = cls.computeRate(distInLogScale, distBegin, yk, d[segmentOfInterest], c[segmentOfInterest], b[segmentOfInterest])
            rate = min(rate, log_rate[segmentOfInterest + 1])
            rate = max(rate, log_rate[segmentOfInterest]) # make sure interpolated rate is within bounds
            arrayRate.append(rate)

        return arrayRate

    # /***
    #  *
    #  * Compute rate from distortion using formula. Use parameters for the curve segment that are passed in.
    #  *
    #  * @param x     distortion for which rate needs to be computed
    #  * @param xk    distortion at the beginning of curve segment
    #  * @param yk    rate (in log scale) at beginning of curve segment
    #  * @param dk    d parameter of relevant curve segment
    #  * @param ck    c parameter of relevant curve segment
    #  * @param bk    b parameter of relevant curve segment
    #  * @return rate in log scale computed using formula and passed in parameters
    #  */
    @staticmethod
    def computeRate(x, xk, yk, dk, ck, bk):

        s = x - xk

        return yk + (s * dk) + (s * s * ck) + (s * s * s * bk)

    @classmethod
    def computeParamsForSegments(cls, rdPointsList, log_rate, log_dist, H, delta, d, c, b, convertRateUsingLogarithm):
        N = len(rdPointsList)

        for i in range(N):
            if (convertRateUsingLogarithm):
                log_rate.append(np.log10(rdPointsList[i][0]))
            else:
                log_rate.append(rdPointsList[i][0])
            log_dist.append(rdPointsList[i][1])

        # // H_i = length of i_th subinterval
        # // delta_i = slope of i_th subinterval
        for i in range(N - 1):
            H.append(log_dist[i + 1] - log_dist[i])
            delta.append((log_rate[i + 1] - log_rate[i]) / H[i])

        # // Determine slopes at each point = coefficient of first-order term

        # // determine slope at starting point
        d.append(cls.pchipend(H[0], H[1], delta[0], delta[1]))

        # // determine slope at all intermediate points: slope is weighted harmonic mean of two slopes (delta_{i-1} and delta_i)
        for i in range(1, N-1):
            d.append((3 * H[i - 1] + 3 * H[i]) / ((2 * H[i] + H[i - 1]) / delta[i - 1] + (H[i] + 2 * H[i - 1]) / delta[i]))

        # // determine slope at end point
        d.append(cls.pchipend(H[N - 2], H[N - 3], delta[N - 2], delta[N - 3]))

        # // determine second- and third-order coefficients for each segment

        for i in range(N - 1):
            c.append((3 * delta[i] - 2 * d[i] - d[i + 1]) / H[i])
            b.append((d[i] - 2 * delta[i] + d[i + 1]) / (H[i] * H[i]))

    @staticmethod
    def pchipend(h1, h2, delta1, delta2):
        # // one-sided formula for endpoints
        # // non-centered shape-preserving three point formula
        d = ((2 * h1 + h2) * delta1 - h1 * delta2) / (h1 + h2)

        # // opposite slopes -> set slope to zero
        if d * delta1 < 0:
            d = 0
        elif (delta1 * delta2 < 0) and (abs(d) > abs(3 * delta1)):
            d = 3 * delta1

        return d
