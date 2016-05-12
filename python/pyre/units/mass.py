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

from SI import kilogram
from SI import kilo, centi, milli

#
# Definitions of common mass units
# Data taken from Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#     fourth edition, John Willey and Sons, 1993

gram = kilogram / kilo
centigram = centi * gram
milligram = milli * gram


# aliases

kg = kilogram
g = gram
cg = centigram
mg = milligram


# other

metric_ton = 1000 * kilogram

ounce = 28.35 * gram
pound = 16 * ounce
ton = 2000 * pound


# version
__id__ = "$Id: mass.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#
# End of file
