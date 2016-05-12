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


class Integer(Column):


    def type(self):
        return "integer"


    def __init__(self, name, default=0, **kwds):
        Column.__init__(self, name, default, **kwds)
        return


# version
__id__ = "$Id: Integer.py,v 1.2 2008-04-13 05:59:14 aivazis Exp $"

# End of file 
