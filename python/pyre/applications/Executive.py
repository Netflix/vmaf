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


class Executive(object):


    # factories
    def createCommandlineParser(self):
        """create a command line parser"""
        
        import pyre.applications
        return pyre.applications.commandlineParser()


    def createRegistry(self, name=None):
        """create a registry instance to store my configuration"""

        if name is None:
            name = self.name

        import pyre.inventory
        return pyre.inventory.registry(name)


    def createCurator(self, name=None):
        """create a curator to handle the persistent store"""

        if name is None:
            name = self.name

        import pyre.inventory
        curator = pyre.inventory.curator(name)

        return curator


    # configuration
    def processCommandline(self, registry, parser=None):
        """convert the command line arguments to a trait registry"""

        if parser is None:
            parser = self.createCommandlineParser()

        help, unprocessedArguments = parser.parse(registry)    

        return help, unprocessedArguments


    def verifyConfiguration(self, unknownProperties, unknownComponents, mode='strict'):
        """verify that the user input did not contain any typos"""

        if mode == 'relaxed':
            return True
        
        if unknownProperties:
            print " ## unrecognized properties:"
            for key, value, locator in unknownProperties:
                print "    %s <- '%s' from %s" % (key, value, locator)

            self.usage()
            return False

        if mode == 'pedantic' and unknownComponents:
            print ' ## unknown components: %s' % ", ".join(unknownComponents)
            self.usage()
            return False
        
        return True


    def pruneRegistry(self):
        registry = self.registry
        
        for trait in self.inventory.properties():
            name = trait.name
            registry.deleteProperty(name)

        for trait in self.inventory.components():
            for name in trait.aliases:
                registry.extractNode(name)

        return registry


    # the default application action
    def main(self, *args, **kwds):
        return


    # user assistance
    def help(self):
        print 'Please consider writing a help screen for this application'
        return


    def usage(self):
        print 'Please consider writing a usage screen for this application'
        return


# version
__id__ = "$Id: Executive.py,v 1.1.1.1 2006-11-27 00:09:54 aivazis Exp $"

# End of file 
