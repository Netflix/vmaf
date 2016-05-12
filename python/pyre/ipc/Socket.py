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


class Socket(socket.socket):


    def connect(self, address):
        try:
            super(Socket, self).connect(address)
        except socket.error, descriptor:
            host, port = address
            errno, reason = descriptor
            raise self.ConnectionError(host, port, errno, reason)

        return
            


    def __init__(self, type):
        socket.socket.__init__(self, type=type)
        return


    # constants
    from socket import SOCK_DGRAM, SOCK_STREAM


    # local exception class
    
    class ConnectionError(Exception):

        def __init__(self, host, port, errno, reason):
            self.host = host
            self.port = port
            self.errno = errno
            self.reason = reason
            return


        def __str__(self):
            msg = "error %d connecting to '%s', port %d: %s" % (
                self.errno, self.host, self.port, self.reason)
            return msg


# version
__id__ = "$Id: Socket.py,v 1.2 2008-01-31 15:41:00 aivazis Exp $"

# End of file 
