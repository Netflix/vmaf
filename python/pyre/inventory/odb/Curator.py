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


from pyre.odb.fs.Curator import Curator as Base


class Curator(Base):


    def getTraits(self, name, extraDepositories=[], encoding='pml'):
        """load cascade of inventory values for component <name>"""

        # initialize the registry object
        registry = self._registryFactory(name)

        # get the relevant codec
        codec = self.codecs[encoding]
        
        # loop over depositories loading relevant traits
        for traits, locator in self.loadSymbol(
            tag=name,
            codec=codec, address=[name], symbol='inventory', extras=extraDepositories,
            errorHandler=self._recordTraitLookup):

            # update the registry
            target = traits.getFacility(name)
            if target:
                # update the registry
                registry = target.update(registry)
                # record status for this lookup
                self._recordTraitLookup(name, locator, 'success')
            else:
                # record the failure
                self._recordTraitLookup(name, locator, "traits for '%s' not found" % name)

        return registry    


    def retrieveComponent(
        self, name, facility, args=(), encoding='odb', vault=[], extraDepositories=[]):
        """construct a component by locating and invoking a component factory"""

        # get the requested codec
        codec = self.codecs[encoding]

        # create the depository address
        if vault:
            location = vault + [name]
        else:
            location = [name]

        # loop over my depositories looking for appropriate factories
        for factory, locator in self.loadSymbol(
            tag=name,
            codec=codec, address=location, symbol=facility, extras=extraDepositories,
            errorHandler=self._recordComponentLookup):

            if not callable(factory):
                self._recordComponentLookup(
                    name, locator, "factory '%s' found but not callable" % facility)
                continue

            try:
                component = factory(*args)
            except TypeError, message:
                self._recordComponentLookup(
                    name, locator, "error invoking '%s': %s" % (facility, message))
                continue

            # set the locator
            if component:
                component.setLocator(locator)

            # record this request
            self._recordComponentLookup(name, locator, "success")

            return component
                
        # return failure
        import pyre.parsing.locators
        locator = pyre.parsing.locators.simple('not found')
        
        return None


    def config(self, registry):
        import os
        import prefix

        # gain access to the installation defaults
        user = prefix._USER_ROOT
        system = prefix._SYSTEM_ROOT
        local = prefix._LOCAL_ROOT

        # look for an environment variable with additional local directories
        try:
            plist = os.environ["PYTHIA_LOCAL"]
            if plist[0] == '[':
                plist = plist[1:]
            if plist[-1] == ']':
                plist = plist[:-1]
            local += plist.split(',')
        except KeyError:
            pass

        # gain access to the user settings from the command line
        db = registry.extractNode(self._DB_NAME)

        # take care of the "local" directories
        if db:
            spec = db.getProperty('user', None)
            if spec is not None:
                user = spec

            spec = db.getProperty('system', None)
            if spec is not None:
                system = spec
                
            spec = db.getProperty('local', None)
            if spec is not None:
                if spec[0] == '[':
                    spec = spec[1:]
                if spec[-1] == ']':
                    spec = spec[:-1]
                local += spec.split(',')

        # add the local depositories to the list
        self.addDepositories(*local)

        # create the root depositories for the system and user areas
        userDepository = self.setUserDepository(user)
        systemDepository = self.setSystemDepository(system)

        return


    def createPrivateDepositories(self, name):
        """ create private system and user depositories from <name>"""

        # initialize the depository list
        depositories = []

        # construct the depositories
        # first the user specific one
        userRoot = self.userDepository
        if userRoot:
            user = userRoot.createDepository(name)
            if user:
                depositories.append(user)

        # next the system wide one
        systemRoot = self.systemDepository
        if systemRoot:
            system = systemRoot.createDepository(name)
            if system:
                depositories.append(system)

        return depositories


    def setUserDepository(self, directory):
        self.userDepository = self.createDepository(directory)
        return self.userDepository


    def setSystemDepository(self, directory):
        self.systemDepository = self.createDepository(directory)
        return self.systemDepository


    def dump(self, extras=None):
        print "curator info:"
        print "    depositories:", [d.name for d in self.depositories]

        if extras:
            print "    local depositories:", [d.name for d in extras]

        if self._traitRequests:
            print "    trait requests:"
            for trait, record in self._traitRequests.iteritems():
                print "        trait='%s'" % trait
                for entry in record:
                    print "            %s: %s" % entry

        if self._componentRequests:
            print "    component requests:"
            for trait, record in self._componentRequests.iteritems():
                print "        component='%s'" % trait
                for entry in record:
                    print "            %s: %s" % entry
            
        return


    def __init__(self, name):
        Base.__init__(self, name)

        # the top level system and user depositories
        self.userDepository = None
        self.systemDepository = None

        # install the peristent store recognizers
        self._registerCodecs()

        # keep a record of requests
        self._traitRequests = {}
        self._componentRequests = {}
        
        # constants
        # the curator commandline argument name
        self._DB_NAME = "inventory"

        return


    def _registerCodecs(self):
        # codec for properties
        import pyre.inventory
        pml = pyre.inventory.codecPML()

        import pyre.odb
        odb = pyre.odb.odb()

        self.registerCodecs(pml, odb)

        return


    def _registryFactory(self, name):
        from Registry import Registry
        return Registry(name)


    def _recordTraitLookup(self, symbol, filename, message):
        requests = self._traitRequests.setdefault(symbol, [])
        requests.append((filename, message))
        return


    def _recordComponentLookup(self, symbol, filename, message):
        requests = self._componentRequests.setdefault(symbol, [])
        requests.append((filename, message))
        return


# version
__id__ = "$Id: Curator.py,v 1.1.1.1 2006-11-27 00:10:02 aivazis Exp $"

# End of file 
