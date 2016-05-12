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

from Primitive import Primitive


class Cone(Primitive):


    def identify(self, visitor):
        return visitor.onCone(self)


    def __init__(self, top, bottom, height):
        self.top = top
        self.bottom = bottom
        self.height = height

        self._info.log("new %s" % self)
                 
        return


    def __str__(self):
        return "cone: top=%s, bottom=%s, height=%s" % (self.top, self.bottom, self.height)


# version
__id__ = "$Id: Cone.py,v 1.1.1.1 2006-11-27 00:09:58 aivazis Exp $"

#
# End of file
