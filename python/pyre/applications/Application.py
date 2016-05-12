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


from pyre.components.Component import Component
from Executive import Executive


class Application(Component, Executive):


    class Inventory(Component.Inventory):

        import pyre.inventory

        typos = pyre.inventory.str(
            name='typos', default='strict',
            validator=pyre.inventory.choice(['relaxed', 'strict', 'pedantic']))
        typos.meta['tip'] = 'specifies the handling of typos in the names of properties and facilities'

        import pyre.weaver
        weaver = pyre.inventory.facility("weaver", factory=pyre.weaver.weaver)
        weaver.meta['tip'] = 'the pretty printer of my configuration as an XML document'

        import journal
        journal = journal.facility()
        journal.meta['tip'] = 'the logging facility'

        dumpconfiguration = pyre.inventory.bool( 'dumpconfiguration', default = 0 )
        dumpconfiguration.meta['tip'] = 'If set, dump configuration to a pml file'
        dumpconfiguration.meta['opacity'] = 100000


    def run(self, *args, **kwds):

        # build storage for the user input
        registry = self.createRegistry()
        self.registry = registry

        # command line
        help, self.argv = self.processCommandline(registry)

        # curator
        curator = self.createCurator()
        self.initializeCurator(curator, registry)

        # look for my settings
        self.initializeConfiguration()

        # give descendants an opportunity to collect input from other (unregistered) sources
        self.collectUserInput(registry)

        # update user options from the command line
        self.updateConfiguration(registry)

        # transfer user input to my inventory
        unknownProperties, unknownComponents = self.applyConfiguration()

        # initialize the trait cascade
        self.init()

        # print a startup page
        self.generateBanner()

        # dump configuration
        if self.inventory.dumpconfiguration: self._saveConfiguration()

        # the main application behavior
        if help:
            self.help()
        elif self._showHelpOnly:
            pass
        elif self.verifyConfiguration(unknownProperties, unknownComponents, self.inventory.typos):
            self.execute(*args, **kwds)

        # shutdown
        self.fini()

        return


    def initializeCurator(self, curator, registry):
        if registry is not None:
            curator.config(registry)
            
        # install the curator
        self.setCurator(curator)

        # adjust the depositories
        # first, register the application specific depository
        curator.depositories += self.inventory.getDepositories()
        # then, any extras specified by my descendants
        curator.addDepositories(*self._getPrivateDepositoryLocations())

        return curator


    def collectUserInput(self, registry):
        """collect user input from additional sources"""
        return


    def generateBanner(self):
        """print a startup screen"""
        return


    def __init__(self, name, facility=None):
        if facility is None:
            facility = "application"
            
        Component.__init__(self, name, facility)
        Executive.__init__(self)
    
        # my name as seen by the shell
        import sys
        self.filename = sys.argv[0]

        # commandline arguments left over after parsing
        self.argv = []

        # the user input
        self.registry = None

        # the code generator
        self.weaver = None

        return


    def _init(self):
        Component._init(self)
        self.weaver = self.inventory.weaver

        renderer = self.getCurator().codecs['pml'].renderer
        self.weaver.renderer = renderer

        return


    def _saveConfiguration(self):
        registry = self.createRegistry()
        registry = retrieveConfiguration( self.inventory, registry )
        stream = open( '%s.pml' % self.name, 'w' )
        self.weaver.weave( registry, stream )
        return


    def _getPrivateDepositoryLocations(self):
        return []




def retrieveConfiguration(inventory, registry, excludes = None):
    """place the current inventory configuration in the given registry"""

    if excludes is None:
        excludes = [
            'weaver',
            'typos',
            'help-properties', 'help', 'help-persistence', 'help-components']

    from pyre.inventory.Facility import Facility
    from pyre.inventory.Property import Property
    from journal.components.Journal import Journal

    node = registry.getNode(inventory._priv_name)

    for prop in inventory.properties():

        name = prop.name
        descriptor = inventory.getTraitDescriptor(name)
        value = descriptor.value
        locator = descriptor.locator

        if name in excludes: continue
        #if isinstance(prop, Property) and value == prop.default: continue
        if isinstance(prop, Facility) and isinstance(value, Journal): continue
        if value and isinstance(prop, Facility): value = value.name

        node.setProperty(name, value, locator)
        continue

    for fac in inventory.facilities():
        name = fac.name
        if name in excludes: continue
        component = fac.__get__(inventory)
        if isinstance(component, Journal): continue
        if component is None:
            raise RuntimeError, "Unable to retrieve component for facility %s" % fac.name
        retrieveConfiguration(component.inventory, node, excludes = excludes)
        continue

    return registry




# version
__id__ = "$Id: Application.py,v 1.1.1.1 2006-11-27 00:09:54 aivazis Exp $"

# End of file 
