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

from pyre.weaver.mills.XMLMill import XMLMill


class Renderer(XMLMill):


    def render(self, inventory):
        document = self.weave(inventory)
        return document


    # handlers

    def onInventory(self, inventory):
        self._rep += ['', '<!DOCTYPE inventory>', '', '<inventory>']

        for facility in inventory.facilities.itervalues():
            facility.identify(self)

        self._rep += ['</inventory>']
        return

    
    def onRegistry(self, registry):

        # bail out of empty registries
        if not registry.properties and not registry.facilities:
            return
        
        self._indent()
        self._write('')
        self._write('<component name="%s">' % registry.name)

        self._indent()
        for trait in registry.properties:
            value = registry.getProperty(trait)
            if trait in registry.facilities:
                self._write('<facility name="%s">%s</facility>' % (trait, value))
            else:
                self._write('<property name="%s">%s</property>' % (trait, value))
                
        self._outdent()

        for facility in registry.facilities:
            component = registry.getFacility(facility)
            if component:
                component.identify(self)

        self._write('</component>')
        self._outdent()
        self._write('')

        return


    def __init__(self):
        XMLMill.__init__(self)
        return


    def _renderDocument(self, document):
        return document.identify(self)
    

# version
__id__ = "$Id: Renderer.py,v 1.1.1.1 2006-11-27 00:10:02 aivazis Exp $"

# End of file 
