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


class NetRenderer(object):


    def render(self, entry):
        entry.meta["ip"] = self._localip
        entry.meta["host"] = self._localhost
        return entry


    def __init__(self):
        import socket

        self._localhost = socket.getfqdn()
        self._localip = socket.gethostbyname(self._localhost)

        return
            

# version
__id__ = "$Id: NetRenderer.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file 
