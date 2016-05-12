#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from math import pi
from SI import radian

degree = pi/180 * radian
arcminute = degree / 60
arcsecond = arcminute / 60

deg = degree
rad = radian

# version
__id__ = "$Id: angle.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#
# End of file
