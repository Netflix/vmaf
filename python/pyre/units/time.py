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

from SI import second
from SI import pico, nano, micro, milli


#
# Definitions of common time units
# Data taken from Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#     fourth edition, John Willey and Sons, 1993

picosecond = pico*second
nanosecond = nano*second
microsecond = micro*second
millisecond = milli*second

# aliases

s = second
ps = picosecond
ns = nanosecond
us = microsecond
ms = millisecond

# other common units

minute = 60 * second
hour = 60 * minute
day = 24 * hour
year = 365.25 * day


# version
__id__ = "$Id: time.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#
# End of file
