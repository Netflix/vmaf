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


class Rotation(Transformation):


    def identify(self, visitor):
        return visitor.onRotation(self)


    def __init__(self, body, vector, angle):
        Transformation.__init__(self, body)

        self.angle = angle
        self.vector = tuple(vector)

        self._info.log(str(self))

        return


    def __str__(self):
        return "rotation: body={%s}, vector=%r, angle=%s" % (self.body, self.vector, self.angle)


# version
__id__ = "$Id: Rotation.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

#
# End of file
