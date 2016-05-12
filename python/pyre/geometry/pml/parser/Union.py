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
import pyre.geometry.operations


class Union(Binary):

    tag = "union"


    def notify(self, parent):
        if not self._b1 or not self._b2:
            raise ValueError("'%s' requires exactly two children" % self.tag)

        union = pyre.geometry.operations.unite(self._b1, self._b2)
        parent.onUnion(union)
        return


# version
__id__ = "$Id: Union.py,v 1.1.1.1 2006-11-27 00:09:58 aivazis Exp $"

# End of file
