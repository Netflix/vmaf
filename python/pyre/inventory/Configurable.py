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


from pyre.parsing.locators.Traceable import Traceable


class Configurable(Traceable):


    # lifecycle management
    def init(self):
        """load user input, initialize my subcomponents and call the custom initialization hook"""

        # initialize my subcomponents
        self.inventory.init()

        # perform any last initializations
        self._init()
        
        return


    def fini(self):
        """call the custom finalization hook and then shut down my subcomponents"""
        
        self._fini()
        self.inventory.fini()
        
        return


    # configuration management
    def retrieveConfiguration(self, registry=None):
        """place my current configuration in the given registry"""

        if registry is None:
            registry = self.createRegistry()

        return self.inventory.retrieveConfiguration(registry)


    def initializeConfiguration(self):
        """initialize my private registry using my private settings"""
        return self.inventory.initializeConfiguration()


    def loadConfiguration(self, filename):
        """open the given filename and retrieve registry settings for me"""
        return self.inventory.loadConfiguration(filename)


    def updateConfiguration(self, registry):
        """load the user setting in <registry> into my inventory"""
        return self.inventory.updateConfiguration(registry)


    def applyConfiguration(self):
        """transfer user settings to my inventory"""

        # apply user settings to my properties
        up, uc = self.configureProperties()
        unknownProperties = up
        unknownComponents = uc

        # apply user settings to my components
        up, uc = self.configureComponents()
        unknownProperties += up
        unknownComponents += uc

        # give descendants a chance to adjust to configuration changes
        self._configure()
        
        return (unknownProperties, unknownComponents)


    def configureProperties(self):
        """set the values of all the properties and facilities in my inventory"""
        up, uc = self.inventory.configureProperties()
        return self._claim(up, uc)


    def configureComponents(self):
        """guide my subcomponents through the configuration process"""
        up, uc = self.inventory.configureComponents()
        return self._claim(up, uc)


    def getDepositories(self):
        return self.inventory.getDepositories()

    # single component management
    def retrieveComponent(self, name, factory, args=(), encoding='odb', vault=[], extras=[]):
        """retrieve component <name> from the persistent store"""
        return self.inventory.retrieveComponent(name, factory, args, encoding, vault, extras)


    def configureComponent(self, component, registry=None):
        """guide <component> through the configuration process"""
        up, uc = self.inventory.configureComponent(component, registry)
        return up, uc


    # curator accessors
    def getCurator(self):
        """return my persistent store manager"""
        return self.inventory.getCurator()


    def setCurator(self, curator):
        """set my persistent store manager"""
        return self.inventory.setCurator(curator)


    # accessors for the inventory items by category
    def properties(self):
        """return a list of all the property objects in my inventory"""
        return self.inventory.properties()


    def propertyNames(self):
        """return a list of the names of all my properties"""
        return self.inventory.propertyNames()


    def facilities(self):
        """return a list of all the facility objects in my inventory"""
        return self.inventory.facilities()

        
    def components(self):
        """return a list of all the components in my inventory"""
        return self.inventory.components()


    # access to trait records, values and descriptors by name
    # used by clients that obtain a listing of these names
    # and want to access the underlying objects
    def getTrait(self, name):
        return self.inventory.getTrait(name)


    def getTraitValue(self, name):
        try:
            return self.inventory.getTraitValue(name)
        except KeyError:
            pass

        raise AttributeError("object '%s' of type '%s' has no attribute '%s'" % (
            self.name, self.__class__.__name__, name))
        

    def getTraitDescriptor(self, name):
        try:
            return self.inventory.getTraitDescriptor(name)
        except KeyError:
            pass

        raise AttributeError("object '%s' of type '%s' has no attribute '%s'" % (
            self.name, self.__class__.__name__, name))


    # support for the help facility
    def showProperties(self):
        """print a report describing my properties"""
        facilityNames = self.inventory.facilityNames()
        propertyNames = self.inventory.propertyNames()
        propertyNames.sort()
        
        print "properties of %r:" % self.name
        for name in propertyNames:
            if name in facilityNames:
                continue
            
            # get the trait object
            trait = self.inventory.getTrait(name)
            # get the common trait attributes
            traitType = trait.type
            default = trait.default
            meta = trait.meta
            validator = trait.validator
            try:
                tip = meta['tip']
            except KeyError:
                tip = '(no documentation available)'

            # get the trait descriptor from the instance
            descriptor = self.inventory.getTraitDescriptor(name)
            # extract the instance specific values
            value = descriptor.value
            locator = descriptor.locator

            print "    %s=<%s>: %s" % (name, traitType, tip)
            print "        default value: %r" % (default,)
            print "        current value: %r, from %s" % (value, locator)
            if validator:
                print "        validator: %s" % validator

        return


    def showComponents(self):
        facilityNames = self.inventory.facilityNames()
        facilityNames.sort()

        print "facilities of %r:" % self.name
        for name in facilityNames:

            # get the facility object
            facility = self.inventory.getTrait(name)
            meta = facility.meta
            try:
                tip = meta['tip']
            except KeyError:
                tip = '(no documentation available)'

            # get the trait descriptor from the instance
            descriptor = self.inventory.getTraitDescriptor(name)
            # extract the instance specific values
            value = descriptor.value
            locator = descriptor.locator

            print "    %s=<component name>: %s" % (name, tip)
            print "        current value: %r, from %s" % (value.name, locator)
            print "        configurable as: %s" % ", ".join(value.aliases)

        return


    def showUsage(self):
        """print a high level usage screen"""
        propertyNames = self.inventory.propertyNames()
        propertyNames.sort()
        facilityNames = self.inventory.facilityNames()
        facilityNames.sort()

        print "component %r" % self.name

        if propertyNames:
            print "    properties:", ", ".join(propertyNames)

        if facilityNames:
            print "    facilities:", ",".join(facilityNames)

        print "For more information:"
        print "  --help-properties: prints details about user settable properties"
        print "  --help-components: prints details about user settable facilities and components"

        return


    def showCurator(self):
        """print a description of the manager of my persistence store"""
        self.inventory.dumpCurator()
        return


    # default implementations of the various factories
    def createRegistry(self, name=None):
        """create a registry instance to store my configuration"""
        if name is None:
            name = self.name
            
        import pyre.inventory
        return pyre.inventory.registry(name)


    def createInventory(self):
        """create my inventory instance"""
        return self.Inventory(self.name)


    def __init__(self, name):
        Traceable.__init__(self)
        
        self.name = name
        self.inventory = self.createInventory()

        # other names by which I am known for configuration purposes
        self.aliases = [ name ]

        import journal
        self._info = journal.info(name)
        self._debug = journal.debug(name)

        # modify the inventory defaults that were hardwired at compile time
        # gives derived components an opportunity to modify their default behavior
        # from what was inherited from their parent's inventory
        self._defaults()
        
        return


    # default implementations for the lifecycle management hooks
    def _defaults(self):
        """modify the default inventory values"""
        return


    def _configure(self):
        """modify the configuration programmatically"""
        return


    def _init(self):
        """wake up"""
        return


    def _fini(self):
        """all done"""
        return


    # misc
    def _claim(self, up, uc):
        """decorate the missing traits with my name"""
        rup = [ (self.name + '.' + key, value, locator) for key, value, locator in up ]
        ruc = [ self.name + '.' + key for key in uc]
        return rup, ruc
        

    # inventory
    from Inventory import Inventory


# version
__id__ = "$Id: Configurable.py,v 1.1.1.1 2006-11-27 00:10:00 aivazis Exp $"

# End of file 
