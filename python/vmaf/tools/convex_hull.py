from typing import Iterable

from .typing_utils import RdPoint


def cross(o: RdPoint, a: RdPoint, b: RdPoint) -> float:
    """
    2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    Returns a positive value, if OAB makes a counter-clockwise turn,
    negative for clockwise turn, and zero if the points are collinear.
    """
    return (a.rate - o.rate) * (b.metric - o.metric) - (a.metric - o.metric) * (b.rate - o.rate)


def calculate_convex_hull(points: Iterable[RdPoint]) -> list[RdPoint]:
    """
    Computes the convex hull of a set of RdPoints.

    Input: an iterable sequence of RdPoints.
    Output: a list of RdPoints of the convex hull in counter-clockwise order,
    starting from the vertex with the lexicographically smallest coordinates.

    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    For this application only the lower hull is used, so upper hull is not calculated.
    Points that are nonmonotonic (higher rate, lower quality than some previous point) are disregarded.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points), key=lambda point: (point.rate, point.metric))

    # Remove points that are nonmonotonic: higher rate, lower quality
    monotonic_points: list[RdPoint] = []
    for point in points:
        if len(monotonic_points) == 0 or point.metric >= monotonic_points[-1].metric:
            monotonic_points.append(point)

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(monotonic_points) == 0:
        return monotonic_points
    if len(monotonic_points) == 1:
        return monotonic_points

    # Build lower hull
    hull: list[RdPoint] = []
    for p in reversed(monotonic_points):
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull
