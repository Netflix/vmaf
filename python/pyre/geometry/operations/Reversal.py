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

from Transformation import Transformation


class Reversal(Transformation):


    def identify(self, visitor):
        return visitor.onReversal(self)


    def __init__(self, body):
        Transformation.__init__(self, body)

        self._info.log(str(self))

        return


    def __str__(self):
        return "reversal: body={%s}" % self.body



# version
__id__ = "$Id: Reversal.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

#
# End of file
