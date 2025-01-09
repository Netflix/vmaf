"""
BD-rate calculator. Implementation validated against JCTVC-E137.
"""

from __future__ import annotations

__copyright__ = "Copyright 2016-2024, Netflix, Inc."
__license__ = "BSD+Patent"

import math

import numpy as np
from typing import Iterable, Any

from scipy.integrate import trapezoid
from scipy.interpolate import pchip_interpolate  # type: ignore[attr-defined]

from vmaf.tools.convex_hull import calculate_convex_hull
from .exceptions import (
    BdRateNoOverlapException,
    BdRateNonMonotonicException,
    BdRateZeroRateException,
    BdRateNotEnoughPointsException,
)
from .typing_utils import RdPoint

INF_REPLACEMENT = 100.0
NUM_SAMPLES = 100


def calculate_bd_rate(
    metric_set1: Iterable[RdPoint],
    metric_set2: Iterable[RdPoint],
    min_metric: float | None = None,
    max_metric: float | None = None,
    use_convex_hull: bool = False,
    at_perc: float | None = None,
) -> float:
    """
    Calculate the Bjontegaard Delta rate (BD-Rate).

    Bjontegaard's metric calculates the average percentage saving in bitrate
    between two rate-distortion curves. This implementation uses Piecewise
    Cubic Hermite Interpolating Polynomial (PCHIP) and the trapezoid method
    for integration.

    Args:
       metric_set1: Iterable of tuples (bitrate, metric) for the first graph.
       metric_set2: Iterable of tuples (bitrate, metric) for the second graph.
       min_metric: Optional minimum metric value for integration.
       max_metric: Optional maximum metric value for integration.
       use_convex_hull: If set, the BD-rate computation will be performed on the convex hull of the points.
       at_perc: If set, the BD-rate will be calculated only at a percentage [0.0 - 100.0] where the metric range of the two curves overlaps.

    Returns:
       The average savings in bitrate at equal quality. Multiply by 100 to get a percentage.

    Raises:
       BdRateNotEnoughPointsException: If either metric set has fewer than 4 points.
       BdRateNonMonotonicException: If either curve is non-monotonic.
       BdRateZeroRateException: If any rate in the metric sets is zero.
       BdRateNoOverlapException: If there is no overlapping interval for integration.
    """

    if not metric_set1 or not metric_set2:
        raise BdRateNotEnoughPointsException("One or both of the metric sets is empty or null.")

    if use_convex_hull:
        metric_set1 = calculate_convex_hull(metric_set1)
        metric_set2 = calculate_convex_hull(metric_set2)

    if at_perc is not None:
        if at_perc < 0 or at_perc > 100:
            raise ValueError(f"at_perc must be between 0 and 100, but got {at_perc}.")

    # pchip_interpolate requires keys sorted by x axis.
    # x-axis will be our metric, not the bitrate, so sort by metric.
    metric_set1 = sorted(metric_set1, key=lambda p: p.metric)
    metric_set2 = sorted(metric_set2, key=lambda p: p.metric)

    if len(metric_set1) < 4 or len(metric_set2) < 4:
        raise BdRateNotEnoughPointsException("Each metric set must contain at least 4 points.")

    if not _is_curve_monotonic(metric_set1) or not _is_curve_monotonic(metric_set2):
        raise BdRateNonMonotonicException("One or both curves are non-monotonic.")

    if not _rates_are_nonzero(metric_set1) or not _rates_are_nonzero(metric_set2):
        raise BdRateZeroRateException("One or both metric sets contain zero rates.")

    # Pull the log of the rate and clamped metric from metric_sets.
    log_rate1 = [math.log(x.rate) for x in metric_set1]
    metric1 = [INF_REPLACEMENT if x.metric == float("inf") else x.metric for x in metric_set1]
    log_rate2 = [math.log(x.rate) for x in metric_set2]
    metric2 = [INF_REPLACEMENT if x.metric == float("inf") else x.metric for x in metric_set2]

    # Integration interval. This metric only works on the area that's
    # overlapping. Extrapolation of these things is sketchy so we avoid.
    min_int = max(min(metric1), min(metric2))
    if min_metric:
        min_int = max(min_int, min_metric)
    max_int = min(max(metric1), max(metric2))
    if max_metric:
        max_int = min(max_int, max_metric)

    # No overlap means no sensible metric possible.
    if max_int <= min_int:
        raise BdRateNoOverlapException()

    if at_perc is None:
        # Use Piecewise Cubic Hermite Interpolating Polynomial interpolation to
        # create 100 new samples points separated by interval.
        samples, interval = np.linspace(min_int, max_int, num=NUM_SAMPLES, retstep=True)
        v1 = pchip_interpolate(metric1, log_rate1, samples)
        v2 = pchip_interpolate(metric2, log_rate2, samples)

        # Calculate the integral using the trapezoid method on the samples.
        int_v1 = trapezoid(v1, dx=float(interval))
        int_v2 = trapezoid(v2, dx=float(interval))

        # Calculate the average improvement.
        avg_exp_diff = (int_v2 - int_v1) / (max_int - min_int)

        # Exponentiate to undo the logarithms
        return math.exp(avg_exp_diff) - 1

    else:
        at_metric = min_int + (max_int - min_int) * at_perc / 100.0
        v1b: Any = pchip_interpolate(metric1, log_rate1, [at_metric])
        v2b: Any = pchip_interpolate(metric2, log_rate2, [at_metric])

        # Exponentiate to undo the logarithms
        return math.exp(v2b[0] - v1b[0]) - 1


def _is_curve_monotonic(points: list[RdPoint]) -> bool:
    """
    Check if the RD curve is monotonic.

    Args:
        points: List of RD points.

    Returns:
        True if the curve is monotonic, False otherwise.
    """
    return all(point1.rate < point2.rate and point1.metric < point2.metric for point1, point2 in zip(points, points[1:]))


def _rates_are_nonzero(points: Iterable[RdPoint]) -> bool:
    """
    Check if all rates in the RD points are non-zero.

    Args:
        points: List of RD points.

    Returns:
        True if all rates are non-zero, False otherwise.
    """
    return all(point.rate != 0 for point in points)
