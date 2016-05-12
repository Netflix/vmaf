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


class Char(Column):


    def type(self):
        return "character (%d)" % self.length


    def __init__(self, name, length, default="", **kwds):
        Column.__init__(self, name, default, **kwds)
        self.length = length
        return


# version
__id__ = "$Id: Char.py,v 1.3 2008-04-14 04:46:17 aivazis Exp $"

# End of file 
