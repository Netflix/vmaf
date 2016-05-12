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


import pyre.util.range
from pyre.inventory.Property import Property


class Slice(Property):


    def __init__(self, name, default=[], meta=None, validator=None):
        Property.__init__(self, name, "slice", default, meta, validator)
        return


    def _cast(self, value):
        if isinstance(value, basestring):
            try:
                value = pyre.util.range.sequence(value)
            except:
                raise TypeError("property '%s': could not convert '%s' to a slice" % (
                    self.name, value))

        if isinstance(value, list):
            return value
            
        raise TypeError("property '%s': could not convert '%s' to a slice" % (self.name, value))
    

# version
__id__ = "$Id: Slice.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

# End of file 
