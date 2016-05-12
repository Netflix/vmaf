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


from pyre.components.Component import Component


class Session(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory

        port = pyre.inventory.int('port', default=50000)
        host = pyre.inventory.str('host', default='localhost')


    def request(self, command, args=None):
        raise NotImplementedError("class %r must override 'request'" % self.__class__.__name__)


    def __init__(self, name, protocol):
        Component.__init__(self, name, facility='session')

        self.host = ''
        self.port = None
        self.protocol = protocol

        self._connection = None

        return


    def _configure(self):
        Component._configure(self)

        self.host = self.inventory.host
        self.port = self.inventory.port

        return


    def _connect(self):
        import pyre.ipc
        self._connection = pyre.ipc.connection(self.protocol)

        self._info.log(
            "attempting to connect to server at %s:%d" % (self.host, self.port))

        self._connection.connect((self.host, self.port))
        return


# version
__id__ = "$Id: Session.py,v 1.1.1.1 2006-11-27 00:10:06 aivazis Exp $"

# End of file 
