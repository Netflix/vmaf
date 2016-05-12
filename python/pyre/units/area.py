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

from length import meter, centimeter, inch, foot, mile


#
# Definitions of common area units
# Data taken from Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#     fourth edition, John Willey and Sons, 1993

square_meter = meter**2
square_centimeter = centimeter**2

square_foot = foot**2
square_inch = inch**2
square_mile = mile**2

acre = 43560 * square_foot
hectare = 10000 * square_meter

barn = 1e-28 * square_meter

# version
__id__ = "$Id: area.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#
# End of file
