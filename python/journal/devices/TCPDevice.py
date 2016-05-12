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


from Device import Device


class TCPDevice(Device):


    def record(self, entry):
        import pyre.ipc
        connection = pyre.ipc.connection('tcp')

        # attempt to connect
        # if refused, just drop the entry for now
        try:
            connection.connect((self.host, self.port))
        except connection.ConnectionError:
            return

        import journal
        request = journal.request(command="record", args=[self.renderer.render(entry)])

        try:
            self._marshaller.send(request, connection)
            result = self._marshaller.receive(connection)
        except self._marshaller.RequestError:
            return

        return


    def __init__(self, key, port, host=''):
        import socket
        from NetRenderer import NetRenderer

        Device.__init__(self, NetRenderer())

        self.host = host
        self.port = port

        import journal
        self._marshaller = journal.pickler()
        self._marshaller.key = key

        return


# version
__id__ = "$Id: TCPDevice.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file
