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


from AbstractNode import AbstractNode


class Component(AbstractNode):


    tag = "component"


    def notify(self, parent):
        return parent.onComponent(self.component)


    def onComponent(self, component):
        self.component.attachNode(component)
        return


    def onFacility(self, facility):
        self.component.setProperty(facility.name, facility.value, facility.locator)
        return


    def onProperty(self, property):
        self.component.setProperty(property.name, property.value, property.locator)
        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, document)
        name = attributes["name"]

        from pyre.inventory.odb.Registry import Registry
        self.component = Registry(name)
        return
    

# version
__id__ = "$Id: Component.py,v 1.1.1.1 2006-11-27 00:10:02 aivazis Exp $"

# End of file 
