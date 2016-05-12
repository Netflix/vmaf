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

from SI import pascal, kilo, mega, giga

#
# Definitions of common pressure units
#
# Data taken from
#     Appendix F of Halliday, Resnick, Walker, "Fundamentals of Physics",
#         fourth edition, John Willey and Sons, 1993
#
#     The NIST Reference on Constants, Units and Uncertainty,
#         http://physics.nist.gov/cuu
#


# aliases

Pa = pascal
kPa = kilo*pascal
MPa = mega*pascal
GPa = giga*pascal


# others

bar = 1e5 * pascal
millibar = 100 * pascal

torr = 133.3 * pascal
atmosphere = 101325 * pascal

atm = atmosphere


# version
__id__ = "$Id: pressure.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#
# End of file
