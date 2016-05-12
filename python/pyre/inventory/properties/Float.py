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


class Float(Property):


    def __init__(self, name, default=0.0, meta=None, validator=None):
        Property.__init__(self, name, "float", default, meta, validator)
        return


    def _cast(self, value):
        return float(value)
    

# version
__id__ = "$Id: Float.py,v 1.1.1.1 2006-11-27 00:10:02 aivazis Exp $"

# End of file 
