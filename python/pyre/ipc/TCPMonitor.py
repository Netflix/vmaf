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

from TCPSocket import TCPSocket
from PortMonitor import PortMonitor


class TCPMonitor(PortMonitor, TCPSocket):


    def install(self, port, maxPort=None, backlog=None):
        PortMonitor.install(self, port, maxPort)

        if backlog is None:
            backlog = self.MAX_QUEUE
            
        self.listen(backlog)

        return

    
    def __init__(self):
        PortMonitor.__init__(self)
        TCPSocket.__init__(self)
        return


    MAX_QUEUE = 10


# version
__id__ = "$Id: TCPMonitor.py,v 1.1.1.1 2006-11-27 00:10:04 aivazis Exp $"

# End of file
