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


import pickle
from pyre.services.TCPSession import TCPSession


class IDDSession(TCPSession):


    class Inventory(TCPSession.Inventory):

        import pyre.inventory

        marshaller = pyre.inventory.facility("marshaller", factory=pyre.idd.pickler)


    def token(self):

        token = self.request(command="token")

        self._info.line("token received:")
        self._info.line("    id: %s" % token.tid)
        self._info.line("    date: %s" % token.date)
        self._info.log("    locator: %s" % token.locator)

        return token


    def __init__(self, name=None):
        if name is None:
            name = "idd-session"

        TCPSession.__init__(self, name)

        self.marshaller = None

        return

        
    def _configure(self):
        TCPSession._configure(self)
        self.marshaller = self.inventory.marshaller
        return


# version
__id__ = "$Id: IDDSession.py,v 1.1.1.1 2006-11-27 00:09:59 aivazis Exp $"

# End of file 
