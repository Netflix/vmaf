#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# journal

def journal():
    global _theJournal
    if _theJournal is not None:
        return _theJournal
    
    from Journal import Journal
    _theJournal = Journal("journal")
    return _theJournal


# facilities and components

def facility(default=None):
    from components.JournalFacility import JournalFacility
    return JournalFacility(default)


# channels

def firewall(name):
    from diagnostics.Firewall import Firewall
    return Firewall().diagnostic(name)


def debug(name):
    from diagnostics.Debug import Debug
    return Debug().diagnostic(name)


def info(name):
    from diagnostics.Info import Info
    return Info().diagnostic(name)


def warning(name):
    from diagnostics.Warning import Warning
    return Warning().diagnostic(name)


def error(name):
    from diagnostics.Error import Error
    return Error().diagnostic(name)


# indices

def firewallIndex():
    from diagnostics.Firewall import Firewall
    return Firewall()


def debugIndex():
    from diagnostics.Debug import Debug
    return Debug()


def infoIndex():
    from diagnostics.Info import Info
    return Info()


def warningIndex():
    from diagnostics.Warning import Warning
    return Warning()


def errorIndex():
    from diagnostics.Error import Error
    return Error()


# register known severities

def register():
    firewallIndex()
    debugIndex()
    infoIndex()
    warningIndex()
    errorIndex()

    return


# devices

def logfile(stream):
    from devices.File import File
    device = File(stream)

    journal().device = device
    return device


def remote(key, port, host="localhost", protocol="tcp"):

    if protocol == "tcp":
        from devices.TCPDevice import TCPDevice
        device = TCPDevice(key, port, host)
    elif protocol == "udp":
        from devices.UDPDevice import UDPDevice
        device = UDPDevice(key, port, host)
    else:
        error('journal').log("unknown protocol '%s'" % protocol)
        return
        
    journal().device = device
    return device


# special setups

def daemon(name=None):
    from services.Daemon import Daemon
    return Daemon(name)


def request(command, args):
    from pyre.services.ServiceRequest import ServiceRequest
    return ServiceRequest(command, args)


def service(name=None):
    from services.JournalService import JournalService
    return JournalService(name)
    

def pickler(name=None):
    if name is None:
        name = "journal-pickler"
        
    from services.Pickler import Pickler
    return Pickler(name)


# misc

def copyright():
    return "journal: Copyright (c) 1998-2005 Michael A.G. Aivazis";


# statics
_theJournal = None

# initialize
try:
    #print " ** __init__.py: importing _journal"
    import _journal
except ImportError:
    hasProxy = False
    msg = info("journal")
    msg.line("could not import the C++ bindings for journal")
    msg.log("control of diagnostics from extension modules is unavailable")
else:
    #print " ** __init__.py: initializing C++ bindings"
    _journal.initialize(journal())
    hasProxy = True

# register the known indices
register()
        
# version
__version__ = "0.8"
__id__ = "$Id: __init__.py,v 1.2 2008-04-13 03:59:03 aivazis Exp $"

#  End of file 
