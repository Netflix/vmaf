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


from pyre.applications.ServiceDaemon import ServiceDaemon


class Daemon(ServiceDaemon):


    class Inventory(ServiceDaemon.Inventory):

        import pyre.inventory

        name = pyre.inventory.str('name', default='idd')


    def createComponent(self):
        name = self.inventory.name

        # instantiate the service
        import pyre.idd
        service = pyre.idd.service(self.inventory.name)

        # register a weaver
        service.weaver = self.weaver

        return service


    def __init__(self, name=None):
        if name is None:
            name = 'idd-harness'
            
        ServiceDaemon.__init__(self, name)

        return


# version
__id__ = "$Id: Daemon.py,v 1.1.1.1 2006-11-27 00:09:59 aivazis Exp $"

# End of file 
