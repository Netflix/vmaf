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


class UDPDevice(Device):


    def record(self, entry):
        import journal
        request = journal.request(command="record", args=[self.renderer.render(entry)])
        self._marshaller.send(request, self._connection)
        return


    def __init__(self, key, port, host=''):
        import socket
        from NetRenderer import NetRenderer

        Device.__init__(self, NetRenderer())

        import pyre.ipc
        self._connection = pyre.ipc.connection('udp')
        self._connection.connect((host, port))

        import journal
        self._marshaller = journal.pickler()
        self._marshaller.key = key

        return


# version
__id__ = "$Id: UDPDevice.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file
