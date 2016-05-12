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


class UDPService(Service):


    def onConnectionAttempt(self, selector, monitor):
        self._info.log("detected activity on port %d" % self.port)

        try:
            request = self.marshaller.receive(monitor)
        except ValueError, msg:
            self._info.log("bad request: %s" % msg)
            return True
        except self.marshaller.RequestError, msg:
            self._info.log(msg)
            return True

        self._info.log("got request: command=%r, args=%r" % (request.command, request.args))

        result = self.evaluator.evaluate(self, request.command, request.args)

        self._info.log("got result: %s" % result)

        return True


    def __init__(self, name=None):
        Service.__init__(self, name)
        return


    def _createPortMonitor(self):
        import pyre.ipc
        return pyre.ipc.monitor('udp')



# version
__id__ = "$Id: UDPService.py,v 1.1.1.1 2006-11-27 00:10:06 aivazis Exp $"

# End of file 
