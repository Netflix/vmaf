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

import pyre.geometry.operations
from Transformation import Transformation


class Reversal(Transformation):


    tag = "reversal"


    def notify(self, parent):
        if not self._body:
            raise ValueError("no body specified in '%s'" % self.tag)

        reversal = pyre.geometry.operations.reverse(body=self._body)
        parent.onReversal(reversal)
        return


# version
__id__ = "$Id: Reversal.py,v 1.1.1.1 2006-11-27 00:09:58 aivazis Exp $"

# End of file
