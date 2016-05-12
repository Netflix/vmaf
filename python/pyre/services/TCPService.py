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

from Service import Service


class TCPService(Service):


    def onConnectionAttempt(self, selector, monitor):
        self._debug.log("detected activity on port %d" % self.port)

        socket, address = monitor.accept()
        if not self.validateConnection(address):
            return True

        try:
            request = self.marshaller.receive(socket)
        except ValueError, msg:
            self._debug.log("bad request: %s" % msg)
            return True
        except self.marshaller.RequestError, msg:
            self._info.log(msg)
            return True

        self._info.log("request from [%d@%s]: command=%r, args=%r" % (
            address[1], address[0], request.command, request.args))

        result = self.evaluator.evaluate(self, request.command, request.args)

        self._debug.log("got result: %s" % result)

        try:
            self.marshaller.send(result, socket)
        except self.marshaller.RequestError, msg:
            self._debug.log(msg)

        return True


    def __init__(self, name=None):
        Service.__init__(self, name)
        return


    def _createPortMonitor(self):
        import pyre.ipc
        return pyre.ipc.monitor('tcp')


# version
__id__ = "$Id: TCPService.py,v 1.1.1.1 2006-11-27 00:10:06 aivazis Exp $"

# End of file 
