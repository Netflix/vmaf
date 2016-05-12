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

        name = pyre.inventory.str('name', default='ipa')


    def createComponent(self):
        name = self.inventory.name

        # instantiate the service
        import pyre.ipa
        service = pyre.ipa.service(self.inventory.name)

        # register a weaver
        service.weaver = self.weaver

        return service


    def __init__(self, name=None):
        if name is None:
            name = 'ipa-harness'
            
        ServiceDaemon.__init__(self, name)

        return


# version
__id__ = "$Id: Daemon.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

# End of file 
