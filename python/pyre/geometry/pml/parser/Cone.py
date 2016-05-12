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


class Cone(AbstractNode):

    tag = "cone"


    def notify(self, parent):
        cone = pyre.geometry.solids.cone(
            top=self._topRadius, bottom=self._bottomRadius, height=self._height)

        parent.onCone(cone)

        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, attributes)

        self._topRadius = self._parse(attributes["topRadius"])
        self._bottomRadius = self._parse(attributes["bottomRadius"])

        self._height = self._parse(attributes["height"])

        return


# version
__id__ = "$Id: Cone.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
