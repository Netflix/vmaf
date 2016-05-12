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


from pyre.applications.Script import Script
from DynamicComponentHarness import DynamicComponentHarness


class ServiceHarness(Script, DynamicComponentHarness):


    class Inventory(Script.Inventory):

        import pyre.inventory

        client = pyre.inventory.str('client')


    def main(self, *args, **kwds):
        # harness the service
        service = self.harnessComponent()
        if not service:
            return

        # generate client configuration
        self.generateClientConfiguration(service)

        # enter the indefinite loop waiting for requests
        service.serve()
        
        return


    def configureHarnessedComponent(self, service, curator, registry):
        value = super(ServiceHarness, self).configureHarnessedComponent(service, curator, registry)
        service.weaver = self.weaver
        return value


    def generateClientConfiguration(self, service):
        clientName = self.client
        if not clientName:
            clientName = service.name + '-session'

        registry = self.createRegistry()
        serviceRegistry = registry.getNode(clientName)
        service.generateClientConfiguration(serviceRegistry)

        stream = file(clientName + '.pml', 'w')
        document = self.weaver.render(registry)
        print >> stream, "\n".join(document)
        stream.close()
            
        return


    def getFacilityName(self, registry):
        """return the facility implemented by my harnessed component"""
        return "service"


    def __init__(self, name=None):
        Script.__init__(self, name)
        DynamicComponentHarness.__init__(self)

        self.client = ''

        return
    

    def _defaults(self):
        Script._defaults(self)
        self.inventory.typos = 'relaxed'
        return


    def _configure(self):
        Script._configure(self)
        self.client = self.inventory.client
        return


# version
__id__ = "$Id: ServiceHarness.py,v 1.1.1.1 2006-11-27 00:09:54 aivazis Exp $"

# End of file 
