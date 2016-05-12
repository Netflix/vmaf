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


class Inventory(object):


    def initializeConfiguration(self):
        # load my settings from the persistent store
        # NYI: load options based on my facility as well?
        #      it might be useful when we auto-generate settings for an entire group of clients
        #      whose names may not be known when their configurations are built
        self._priv_registry = self._priv_curator.getTraits(
            self._priv_name, self._priv_depositories)

        return


    def loadConfiguration(self, filestem):
        """load the registry contained in the given pml file (without the extension)"""

        import pyre.inventory
        codec = pyre.inventory.codecPML()
        shelf = codec.open(filestem)

        return shelf['inventory']


    def updateConfiguration(self, registry):
        return self._priv_registry.update(registry)


    def configureProperties(self):
        """configure my properties using user settings in my registry"""
        
        unknownProperties = []

        # loop over the registry property entries and
        # attempt to set the value of the corresponding inventory item
        # print 'pyre.inventory.Inventory.configureProperties', self
        for name, descriptor in self._priv_registry.properties.iteritems():
            # print 'pyre.inventory.Inventory.configureProperties', name, descriptor.value
            try:
                prop = self._traitRegistry[name]
            except KeyError:
                unknownProperties.append((name, descriptor.value, descriptor.locator))
                continue
            prop._set(self, descriptor.value, descriptor.locator)
            # print 'pyre.inventory.Inventory.configureProperties: *******', name, self._getTraitValue(name)

        return unknownProperties, []


    def configureComponents(self):
        """configure my components using options from my registry"""

        unknownProperties = []
        unknownComponents = []

        myComponents = self.components()

        aliases = {}
        for component in myComponents:
            # associate a persistent store with every subcomponent
            component.setCurator(self._priv_curator)
            component.initializeConfiguration()

            # construct a list of the public names of this component
            # setting are overriden from left to right
            componentAliases = list(component.aliases)
            componentAliases.reverse()
            # for each registered public name of this component
            for alias in componentAliases:
                # register this name so we can hunt down typos
                aliases[alias] = component

                registry = self._priv_registry.getFacility(alias)
                if registry:
                    component.updateConfiguration(registry)

            up, uc = component.applyConfiguration()
            unknownProperties += up
            unknownComponents += uc

        # loop over the registry facility entries and
        # update the configuration of all the named components/facilities
        # note that this only affects components for which there are settings in the registry
        # this is done in a separate loop because it provides an easy way to catch typos
        # on the command line
        for name, registry in self._priv_registry.facilities.iteritems():
            try:
                component = aliases[name]
            except KeyError:
                unknownComponents.append(name)
                continue

        return (unknownProperties, unknownComponents)


    def retrieveConfiguration(self, registry):
        """place the current inventory configuration in the given registry"""

        from Facility import Facility
        from Property import Property

        node = registry.getNode(self._priv_name)

        for prop in self._traitRegistry.itervalues():

            name = prop.name
            descriptor = self.getTraitDescriptor(name)
            value = descriptor.value
            locator = descriptor.locator

            if value and isinstance(prop, Facility):
                value = value.name

            node.setProperty(name, value, locator)

        for component in self.components():
            component.retrieveConfiguration(node)
            
        return registry


    def configureComponent(self, component, registry=None):
        """configure <component> using options from the given registry"""

        # if none were given, let the registry be our own
        if registry is None:
            registry = self._priv_registry

        # set the component's curator
        component.setCurator(self._priv_curator)
        component.initializeConfiguration()

        # find any relevant traits in my registry
        # look for facility traits
        aliases = list(component.aliases)
        aliases.reverse()
        for alias in aliases:
            traits = registry.getFacility(alias)
            component.updateConfiguration(traits)

        # apply the settings
        unknownProperties, unknownComponents = component.applyConfiguration()

        # return the unrecognized traits
        return (unknownProperties, unknownComponents)


    def retrieveComponent(
        self, name, factory, args=(), encoding='odb', vault=[], extraDepositories=[]):
        """retrieve component <name> from the persistent store"""

        if extraDepositories:
            import journal
            journal.firewall("inventory").log("non-null extraDepositories")

        return self._priv_curator.retrieveComponent(
            name=name, facility=factory, args=args, encoding=encoding,
            vault=vault, extraDepositories=self._priv_depositories)
        

    def init(self):
        """initialize subcomponents"""

        for component in self.components():
            component.init()

        return


    def fini(self):
        """finalize subcomponents"""

        for component in self.components():
            component.fini()

        return


    # lower level interface
    def getCurator(self):
        """return the curator that resolves my trait requests"""
        return self._priv_curator


    def setCurator(self, curator):
        """set my persistent store manager and initialize my registry"""

        # keep track of the curator
        self._priv_curator = curator

        # construct my private depositories
        self._priv_depositories = self._createDepositories()

        return


    def dumpCurator(self):
        """print a description of the manager of my persistence store"""
        return self._priv_curator.dump(self._priv_depositories)


    def getDepositories(self):
        """return my private depositories"""
        return self._priv_depositories


    def retrieveShelves(self, address, extension):
        return self._priv_curator.retrieveShelves(
            address, extension, extraDepositories=self._priv_depositories)


    def getTraitDescriptor(self, traitName):
        try:
            return self._getTraitDescriptor(traitName)

        except KeyError:
            pass
        
        self._forceInitialization(traitName)
        return self._getTraitDescriptor(traitName)


    def getTraitValue(self, traitName):
        try:
            return self._getTraitValue(traitName)

        except KeyError:
            pass
        
        return self._forceInitialization(traitName)


    def getTrait(self, traitName):
        return self._traitRegistry[traitName]


    # accessors for the inventory items by category
    def properties(self):
        """return a list of my property objects"""
        return self._traitRegistry.values()


    def propertyNames(self):
        """return a list of the names of all my traits"""
        return self._traitRegistry.keys()


    def facilities(self):
        """return a list of my facility objects"""
        return self._facilityRegistry.values()

        
    def facilityNames(self):
        """return a list of the names of all my facilities"""
        return self._facilityRegistry.keys()


    def components(self):
        """return a list of my components"""
        candidates = [
            facility.__get__(self) for facility in self._facilityRegistry.itervalues() ]
        return filter(None, candidates)


    def __init__(self, name):
        # the name of the configurable that manages me
        self._priv_name = name
        
        # the manager of my persistent trait store
        self._priv_curator = None

        # the accumulator of user supplied state
        self._priv_registry = None

        # my private depositories
        self._priv_depositories = []

        # local storage for the descriptors created by the various traits
        self._priv_inventory = {}

        return


    def _createDepositories(self):
        depositories = self._priv_curator.createPrivateDepositories(self._priv_name)
        return depositories
    

    def _getTraitValue(self, name):
        return self._getTraitDescriptor(name).value


    def _setTraitValue(self, name, value, locator):
        descriptor = self._getTraitDescriptor(name)
        descriptor.value = value
        descriptor.locator = locator
        return


    def _initializeTraitValue(self, name, value, locator):
        descriptor = self._createTraitDescriptor()
        descriptor.value = value
        descriptor.locator = locator
        self._setTraitDescriptor(name, descriptor)
        return


    def _createTraitDescriptor(self):
        from Descriptor import Descriptor
        return Descriptor()


    def _getTraitDescriptor(self, name):
        return self._priv_inventory[name]


    def _setTraitDescriptor(self, name, descriptor):
        self._priv_inventory[name] = descriptor
        return


    def _forceInitialization(self, name):
        trait = self._traitRegistry[name]
        return trait.__get__(self)


    # trait registries
    _traitRegistry = {}
    _facilityRegistry = {}


    # metaclass
    from Notary import Notary
    __metaclass__ = Notary


# version
__id__ = "$Id: Inventory.py,v 1.1.1.1 2006-11-27 00:10:00 aivazis Exp $"

# End of file 
