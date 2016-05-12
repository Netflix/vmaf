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


class List(Property):


    def __init__(self, name, default=None, meta=None, validator=None):
        if default is None:
            default = []
        Property.__init__(self, name, "list", default, meta, validator)
        return


    def _cast(self, value):
        if isinstance(value, basestring):
            if value and value[0] in '[({':
                value = value[1:]
            if value and value[-1] in '])}':
                value = value[:-1]

            if not value:
                return []
                
            value = value.split(",")
            return value

        if isinstance(value, list):
            return value
            
        raise TypeError("property '%s': could not convert '%s' to a list" % (self.name, value))
    

# version
__id__ = "$Id: List.py,v 1.3 2007-11-28 09:36:43 aivazis Exp $"

# End of file 
