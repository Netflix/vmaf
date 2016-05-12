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


class Double(Column):


    def type(self):
        return "double precision"


    def __init__(self, name, **kwds):
        Column.__init__(self, name, **kwds)
        return


# version
__id__ = "$Id: Double.py,v 1.1.1.1 2006-11-27 00:09:55 aivazis Exp $"

# End of file 
