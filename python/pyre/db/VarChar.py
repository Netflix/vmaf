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


from Column import Column


class VarChar(Column):


    def type(self):
        return "character varying (%d)" % self.length


    def __init__(self, name, length, default="", **kwds):
        Column.__init__(self, name, default, **kwds)
        self.length = length
        return


# version
__id__ = "$Id: VarChar.py,v 1.2 2008-04-13 05:59:22 aivazis Exp $"

# End of file 
