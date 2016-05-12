#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.inventory.Facility import Facility


class DeviceFacility(Facility):


    def __init__(self, factory=None, args=[]):
        if factory is None:
            args = []
            from Console import Console as factory

        Facility.__init__(self, name="device", factory=factory, args=args)

        return


# version
__id__ = "$Id: DeviceFacility.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file 
