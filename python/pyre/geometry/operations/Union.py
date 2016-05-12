#!/usr/bin/env python
#
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Binary import Binary


class Union(Binary):


    def identify(self, visitor):
        return visitor.onUnion(self)


    def __init__(self, op1, op2):
        Binary.__init__(self, op1, op2)

        self._info.log(str(self))

        return


    def __str__(self):
        return "union: op1={%s}, op2={%s}" % (self.op1, self.op2)


# version
__id__ = "$Id: Union.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

#
# End of file
