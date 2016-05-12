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


class Sphere(AbstractNode):

    tag = "sphere"


    def notify(self, parent):
        sphere = pyre.geometry.solids.sphere(radius=self._radius)
        parent.onSphere(sphere)

        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, attributes)
        self._radius = self._parse(attributes["radius"])
        return


# version
__id__ = "$Id: Sphere.py,v 1.1.1.1 2006-11-27 00:09:58 aivazis Exp $"

# End of file
