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


class VarCharArray(Column):


    def type(self):
        return "character varying(%d)[]" % self.length


    def declaration(self):
        default = self.default
        self.default = None
        ret = Column.declaration(self)
        self.default = default
        return ret


    def __init__(self, name, length, **kwds):
        Column.__init__(self, name, **kwds)
        self.length = length
        return


    def _format(self, value):
        if value is None: value = []
        s = ','.join( value )
        s = '{' + s + '}'
        return s


# version
__id__ = "$Id: VarChar.py,v 1.1.1.1 2006-11-27 00:09:55 aivazis Exp $"

# End of file 
