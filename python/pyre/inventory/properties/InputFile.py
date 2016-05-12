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


class InputFile(Property):


    def __init__(self, name, default=None, meta=None, validator=None):
        if default is None:
            import sys
            default = sys.stdin
            
        Property.__init__(self, name, "file", default, meta, validator)
        return


    def _cast(self, value):
        if isinstance(value, basestring):
            if value == "stdin":
                import sys
                value = sys.stdin
            else:
                value = file(value, "r")
        
        return value


# version
__id__ = "$Id: InputFile.py,v 1.1.1.1 2006-11-27 00:10:02 aivazis Exp $"

# End of file 
