#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.units.SI import joule, kelvin, mole


# Source: physics.nist.gov/constants
#
# Peter J. Mohr and Barry N. Taylor,
#    CODATA Recommended Values of the Fundamental Physical Constants: 1998
#    Journal of Physical and Chemical Reference Data, to be published
#

boltzmann = 1.3806503e-23 * joule/kelvin
avogadro = 6.02214199e23 / mole

gas_constant = 8.314472 * joule/(mole*kelvin)

# aliases

k = boltzmann
N_A = avogadro
L = avogadro
R = gas_constant


# version
__id__ = "$Id: fundamental.py,v 1.1.1.1 2006-11-27 00:09:59 aivazis Exp $"

#
# End of file
