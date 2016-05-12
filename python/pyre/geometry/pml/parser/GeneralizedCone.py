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


class GeneralizedCone(AbstractNode):

    tag = "generalized-cone"


    def notify(self, parent):
        cone = pyre.geometry.solids.generalizedCone(
            major=self._major, minor=self._minor, scale=self._scale,
            height=self._height)

        parent.onCone(cone)

        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, attributes)

        self._major = self._parse(attributes["major"])
        self._minor = self._parse(attributes["minor"])
        self._scale = self._parse(attributes["scale"])
        self._height = self._parse(attributes["height"])

        return


# version
__id__ = "$Id: GeneralizedCone.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
