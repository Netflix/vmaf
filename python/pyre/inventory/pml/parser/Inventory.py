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


class Inventory(AbstractNode):


    tag = "inventory"


    def notify(self, parent):
        parent.onInventory(self.inventory)
        return


    def onComponent(self, component):
        self.inventory.attachNode(component)
        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, document)

        from pyre.inventory.odb.Inventory import Inventory
        self.inventory = Inventory('root')

        return


# version
__id__ = "$Id: Inventory.py,v 1.1.1.1 2006-11-27 00:10:02 aivazis Exp $"

# End of file 
