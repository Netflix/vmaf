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


from pyre.inventory.Property import Property


class String(Property):


    def __init__(self, name, default="", meta=None, validator=None):
        Property.__init__(self, name, "str", default, meta, validator)
        return


    def _cast(self, value):
        return str(value)


# version
__id__ = "$Id: String.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

# End of file 
