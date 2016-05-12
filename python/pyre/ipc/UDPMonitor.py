#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from UDPSocket import UDPSocket
from PortMonitor import PortMonitor


class UDPMonitor(PortMonitor, UDPSocket):


    def __init__(self):
        PortMonitor.__init__(self)
        UDPSocket.__init__(self)
        return


# version
__id__ = "$Id: UDPMonitor.py,v 1.1.1.1 2006-11-27 00:10:04 aivazis Exp $"

# End of file 
