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

from Application import Application
from Daemon import Daemon as Stager
from ComponentHarness import ComponentHarness


class ServiceDaemon(Application, Stager, ComponentHarness):


    class Inventory(Application.Inventory):

        import pyre.inventory

        client = pyre.inventory.str('client')
        home = pyre.inventory.str('home', default='/tmp')


    def main(self, *args, **kwds):
        # harness the service
        idd = self.harnessComponent()
        if not idd:
            return

        # generate client configuration
        self.generateClientConfiguration(idd)

        # enter the indefinite loop waiting for requests
        idd.serve()
        
        return


    def generateClientConfiguration(self, component):
        clientName = self.inventory.client
        if not clientName:
            clientName = component.name + '-session'

        registry = self.createRegistry()
        componentRegistry = registry.getNode(clientName)
        component.generateClientConfiguration(componentRegistry)

        stream = file(clientName + '.pml', 'w')
        document = self.weaver.render(registry)
        print >> stream, "\n".join(document)
        stream.close()
            
        return


    def __init__(self, name):
        Application.__init__(self, name, facility='daemon')
        Stager.__init__(self)
        ComponentHarness.__init__(self)
        return


    def _configure(self):
        Application._configure(self)

        import os
        self.home = os.path.abspath(self.inventory.home)
        return


# version
__id__ = "$Id: ServiceDaemon.py,v 1.1.1.1 2006-11-27 00:09:54 aivazis Exp $"

# End of file 
