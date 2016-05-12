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


from Session import Session


class UDPSession(Session):


    def request(self, command, args=None):
        if args is None:
            args = ()

        import pyre.services
        request = pyre.services.request(command, args)

        self._connect()

        self._info.log("sending request: command=%r" % command)
        self.marshaller.send(request, self._connection)
        self._info.log("request sent")

        return


    def __init__(self, name):
        Session.__init__(self, name, protocol='udp')
        return


# version
__id__ = "$Id: UDPSession.py,v 1.1.1.1 2006-11-27 00:10:06 aivazis Exp $"

# End of file 
