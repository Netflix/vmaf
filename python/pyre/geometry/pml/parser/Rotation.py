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


class Rotation(Transformation):


    tag = "rotation"


    def onAngle(self, angle):
        self._angle = angle
        return


    def onVector(self, vector):
        self._vector = vector
        return


    def notify(self, parent):
        if not self._body:
            raise ValueError("no body specified in '%s'" % self.tag)
        if not self._vector:
            raise ValueError("no vector specified in '%s'" % self.tag)
        if not self._angle:
            raise ValueError("no angle specified in '%s'" % self.tag)

        rotation = pyre.geometry.operations.rotate(
            body=self._body, vector=self._vector, angle=self._angle)

        parent.onRotation(rotation)

        return


    def __init__(self, document, attributes):
        Transformation.__init__(self, attributes)
        self._angle = None
        self._vector = None
        return


# version
__id__ = "$Id: Rotation.py,v 1.1.1.1 2006-11-27 00:09:58 aivazis Exp $"

# End of file
