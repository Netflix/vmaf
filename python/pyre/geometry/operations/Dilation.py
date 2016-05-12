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


class Dilation(Transformation):


    def identify(self, visitor):
        return visitor.onDilation(self)


    def __init__(self, body, scale):
        Transformation.__init__(self, body)
        self.scale = scale

        self._info.log(str(self))
        
        return


    def __str__(self):
        return "dilation: body={%s}, scale=%s" % (self.body, self.scale)


# version
__id__ = "$Id: Dilation.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

#
# End of file
