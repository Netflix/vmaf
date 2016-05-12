#!/usr/bin/env python
#
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                              Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


# body compositions

def intersect(blank, tool):
    from Intersection import Intersection
    return Intersection(blank, tool)


def subtract(blank, tool):
    from Difference import Difference
    return Difference(blank, tool)


def unite(blank, tool):
    from Union import Union
    return Union(blank, tool)


# body transformations

def dilate(body, scale):
    from Dilation import Dilation
    return Dilation(body, scale)


def reflect(body, vector):
    from Reflection import Reflection
    return Reflection(body, vector)


def reverse(body):
    from Reversal import Reversal
    return Reversal(body)


def rotate(body, vector, angle):
    from Rotation import Rotation
    return Rotation(body, vector, angle)


def translate(body, vector):
    from Translation import Translation
    return Translation(body, vector)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

#
# End of file
