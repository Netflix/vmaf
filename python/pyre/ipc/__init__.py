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


def connection(mode):
    if mode == 'tcp':
        from TCPSocket import TCPSocket
        return TCPSocket()

    if mode == 'udp':
        from UDPSocket import UDPSocket
        return UDPSocket()

    import journal
    journal.error('pyre.ipc').log("unknown connection mode '%s'" % mode)

    return None


def monitor(mode):
    if mode == 'tcp':
        from TCPMonitor import TCPMonitor
        return TCPMonitor()

    if mode == 'udp':
        from UDPMonitor import UDPMonitor
        return UDPMonitor()

    import journal
    journal.error('pyre.ipc').log("unknown monitor mode '%s'" % mode)

    return None


def selector():
    from Selector import Selector
    return Selector()

# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:10:04 aivazis Exp $"

# End of file 
