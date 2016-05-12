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


from pyre.inventory.Configurable import Configurable


class Component(Configurable):


    class Inventory(Configurable.Inventory):

        import pyre.inventory

        usage = pyre.inventory.bool("help", default=False)
        usage.meta['tip'] = 'prints a screen that describes my traits'

        showProperties = pyre.inventory.bool("help-properties", default=False)
        showProperties.meta['tip'] = 'prints a screen that describes my properties'

        showComponents = pyre.inventory.bool("help-components", default=False)
        showComponents.meta['tip'] = 'prints a screen that describes my subcomponents'

        showCurator = pyre.inventory.bool("help-persistence", default=False)
        showCurator.meta['tip'] = 'prints a screen that describes my persistent store'


    def updateConfiguration(self, registry):
        # verify that we were handed the correct registry node
        if registry:
            name = registry.name
            if name not in self.aliases:
                import journal
                journal.firewall("inventory").log(
                    "bad registry node: %s != %s" % (name, self.name))

        return Configurable.updateConfiguration(self, registry)


    def __init__(self, name, facility):
        Configurable.__init__(self, name)
        self.facility = facility

        self._showHelpOnly = False
        
        return


    def _init(self):
        Configurable._init(self)

        if self.inventory.usage:
            self.showUsage()
            self._showHelpOnly = True

        if self.inventory.showProperties:
            self.showProperties()
            self._showHelpOnly = True

        if self.inventory.showComponents:
            self.showComponents()
            self._showHelpOnly = True

        if self.inventory.showCurator:
            self.showCurator()
            self._showHelpOnly = True

        if not self._showHelpOnly:
            for component in self.components():
                if component._showHelpOnly:
                    self._showHelpOnly = True
                    break

        return


# version
__id__ = "$Id: Component.py,v 1.1.1.1 2006-11-27 00:09:54 aivazis Exp $"

# End of file 
