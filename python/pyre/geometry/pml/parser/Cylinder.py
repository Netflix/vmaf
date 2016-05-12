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

import pyre.geometry.solids
from AbstractNode import AbstractNode


class Cylinder(AbstractNode):

    tag = "cylinder"


    def notify(self, parent):
        cylinder = pyre.geometry.solids.cylinder(radius=self._radius, height=self._height)
        parent.onCylinder(cylinder)

        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, attributes)
        self._radius = self._parse(attributes["radius"])
        self._height = self._parse(attributes["height"])
        return


# version
__id__ = "$Id: Cylinder.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
