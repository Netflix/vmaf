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


class Real(Column):


    def type(self):
        return "real"


    def __init__(self, name, default=0.0, **kwds):
        Column.__init__(self, name, default, **kwds)
        return


    def _cast(self,value):
        return float(value)


# version
__id__ = "$Id: Real.py,v 1.2 2008-04-13 06:00:17 aivazis Exp $"

# End of file 
