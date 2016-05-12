#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class Registry(object):


    def identify(self, inspector):
        return inspector.onRegistry(self)


    def getFacility(self, name, default=None):
        return self.facilities.get(name, default)


    def getProperty(self, name, default=''):
        try:
            return self.properties[name].value
        except KeyError:
            return default

        # UNREACHABLE
        import journal
        journal.firewall("inventory").log("UNREACHABLE")
        return


    def setProperty(self, name, value, locator):
        self.properties[name] = self._createDescriptor(value, locator)
        return


    def deleteProperty(self, name):
        """remove the named property"""

        try:
            del self.properties[name]
        except KeyError:
            pass
        
        return


    def update(self, registry):
        if not registry:
            return self
        
        for name, descriptor in registry.properties.iteritems():
            # print "pyre.inventory.odb.Registry:", name, descriptor.value, descriptor.locator
            self.setProperty(name, descriptor.value, descriptor.locator)

        for name, node in registry.facilities.iteritems():
            self.getNode(name).update(node)

        return self


    def getNode(self, name):
        try:
            node = self.facilities[name]
        except KeyError:
            node = Registry(name)
            self.facilities[name] = node

        return node


    def attachNode(self, node):
        self.facilities[node.name] = node
        return


    def extractNode(self, facility):
        try:
            node = self.facilities[facility]
        except KeyError:
            return None

        del self.facilities[facility]
        return node


    def render(self):

        listing = [
            ("%s.%s" % (self.name, name), descriptor.value, "%s" % descriptor.locator)
            for name, descriptor in self.properties.iteritems()
            ]

        listing += [
            ("%s.%s" % (self.name, name), value, "%s" % locator)
            for facility in self.facilities.itervalues()
            for name, value, locator in facility.render() 
            ]

        return listing


    def __init__(self, name):
        self.name = name
        self.properties = {}
        self.facilities = {}
        return


    def _createDescriptor(self, value, locator):
        from Descriptor import Descriptor
        return Descriptor(value, locator)


# version
__id__ = "$Id: Registry.py,v 1.1.1.1 2006-11-27 00:10:02 aivazis Exp $"

# End of file 
