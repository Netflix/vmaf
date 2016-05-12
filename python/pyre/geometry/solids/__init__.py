#!/usr/bin/env python
#
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def block(diagonal):
    from Block import Block
    return Block(diagonal)


def cone(top, bottom, height):
    from Cone import Cone
    return Cone(top, bottom, height)


def cylinder(radius, height):
    from Cylinder import Cylinder
    return Cylinder(radius, height)


def generalizedCone(major, minor, scale, height):
    from GeneralizedCone import GeneralizedCone
    return GeneralizedCone(major, minor, scale, height)


def prism():
    # NYI
    pass


def pyramid():
    # NYI
    pass


def sphere(radius):
    from Sphere import Sphere
    return Sphere(radius)


def torus(major, minor):
    from Torus import Torus
    return Torus(major, minor)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:09:59 aivazis Exp $"

#
# End of file
