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


from pyre.xml.Document import Document as DocumentNode


class Document(DocumentNode):


    tags = [
        "Inventory",
        "Component", "Facility", "Property"
        ]


    def onInventory(self, inventory):
        self.document = inventory
        return


# version
__id__ = "$Id: Document.py,v 1.1.1.1 2006-11-27 00:10:02 aivazis Exp $"

# End of file 
