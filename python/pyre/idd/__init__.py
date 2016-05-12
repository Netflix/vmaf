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


def daemon(name=None):
    from Daemon import Daemon
    return Daemon(name)


def service(name=None):
    from IDDService import IDDService
    return IDDService(name=None)


def recordLocator():
    from RecordLocator import RecordLocator
    return RecordLocator()


def session(name=None):
    from IDDSession import IDDSession
    return IDDSession(name)


def pickler(name=None):
    if name is None:
        name = 'idd-pickler'

    from Pickler import Pickler
    return Pickler(name)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:10:00 aivazis Exp $"

# End of file 
