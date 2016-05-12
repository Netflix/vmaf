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


class IPASession(TCPSession):


    class Inventory(TCPSession.Inventory):

        import pyre.inventory

        marshaller = pyre.inventory.facility("marshaller", factory=pyre.ipa.pickler)


    def login(self, username, password):
        self._info.log("login request for user '%s'" % username)
        request = self.request(command='login', args=[username, password])
        return request


    def refresh(self, username, ticket):
        self._info.log("ticketed request for user '%s':'%s'" % (username, ticket))
        request = self.request(command='refresh', args=[username, ticket])
        return request


    def logout(self, username, ticket):
        self._info.log("logout request for user '%s':'%s'" % (username, ticket))
        request = self.request(command='logout', args=[username, ticket])
        return request


    def __init__(self, name=None):
        if name is None:
            name = "ipa-session"

        TCPSession.__init__(self, name)

        self.marshaller = None

        return


    def _configure(self):
        TCPSession._configure(self)
        self.marshaller = self.inventory.marshaller
        return


# version
__id__ = "$Id: IPASession.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

# End of file 
