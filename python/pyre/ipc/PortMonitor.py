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


import socket


class PortMonitor(object):


    def install(self, port, maxPort=None):
        if maxPort is None:
            maxPort = self.MAX_PORT
            
        if port < self.MIN_PORT or port > maxPort:
            msg = "requested port %r is outside the range [%r, %r]" % (
                port, self.MIN_PORT, maxPort)
            raise ValueError(msg)

        minPort = port
        while port <= maxPort:
            try:
                self.bind(('', port))
                self.port = port
                self._debug.log("successfully installed at port %d" % self.port)
                return

            except socket.error, error:
                number, message = error
                self._debug.log(
                    "failed to activate server at port %d: error %d: %s" % (port, number, message))

            port += 1
            
        # no available ports in the range
        msg = "no ports available in the range [%d, %d]" % (minPort, maxPort)
        raise ValueError, msg

        
    def __init__(self):
        self.port = None

        import journal
        self._debug = journal.debug("pyre.ipc.monitor")

        return


    # constants
    MIN_PORT = 1024
    MAX_PORT = 64*1024 - 1


# version
__id__ = "$Id: PortMonitor.py,v 1.1.1.1 2006-11-27 00:10:04 aivazis Exp $"

# End of file 
