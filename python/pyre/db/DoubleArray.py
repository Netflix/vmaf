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


class DoubleArray(Column):


    def type(self):
        return "float8[]"


    def declaration(self):
        default = self.default
        self.default = None
        ret = Column.declaration(self)
        self.default = default
        return ret


    def __init__(self, name, **kwds):
        Column.__init__(self, name, **kwds)
        return


    def _cast(self, value):
        if isinstance(value, str): value = eval( value )
        if isinstance(value, list) or isinstance(value, tuple):
            for item in value:
                assert isinstance(item, float) or isinstance(item, int)
                continue
            return value
        raise NotImplementedError


    def _format(self, value):
        s = str(value)
        s = '{' + s[1:-1] + '}'
        return s


# version
__id__ = "$Id: Double.py,v 1.1.1.1 2006-11-27 00:09:55 aivazis Exp $"

# End of file 
