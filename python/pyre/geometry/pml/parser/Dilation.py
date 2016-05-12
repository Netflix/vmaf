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


class Dilation(Transformation):

    tag = "dilation"


    def onScale(self, scale):
        self._scale = scale
        return


    def notify(self, parent):
        if not self._body:
            raise ValueError("no body specified in '%s'" % self.tag)
        if not self._scale:
            raise ValueError("no scale specified in '%s'" % self.tag)

        dilation = pyre.geometry.operations.dilate(body=self._body, scale=self._scale)
        parent.onDilation(dilation)
        return


    def __init__(self, document, attributes):
        Transformation.__init__(self, attributes)
        self._scale = None
        return


# version
__id__ = "$Id: Dilation.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
