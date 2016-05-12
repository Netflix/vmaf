#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

from pyre.applications.ServiceDaemon import ServiceDaemon


class Daemon(ServiceDaemon):


    def createComponent(self):
        from JournalService import JournalService
        component = JournalService()
        return component
        

    def __init__(self, name=None):
        if name is None:
            name = 'journald-harness'
            
        ServiceDaemon.__init__(self, name)

        return

# version
__id__ = "$Id: Daemon.py,v 1.1.1.1 2006-11-27 00:09:36 aivazis Exp $"

#  End of file 
