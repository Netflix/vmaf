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


class Difference(Binary):

    tag = "difference"


    def notify(self, parent):
        if not self._b1 or not self._b2:
            raise ValueError("'%s' requires exactly two children" % self.tag)
        
        difference = pyre.geometry.operations.subtract(self._b1, self._b2)
        parent.onDifference(difference)

        return


# version
__id__ = "$Id: Difference.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
