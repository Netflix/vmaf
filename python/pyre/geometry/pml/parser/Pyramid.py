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


class Pyramid(AbstractNode):

    tag = "pyramid"


    def notify(self, parent):
        pyramid = pyre.geometry.solids.pyramid()
        parent.onPyramid(pyramid)

        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, attributes)
        return


# version
__id__ = "$Id: Pyramid.py,v 1.1.1.1 2006-11-27 00:09:58 aivazis Exp $"

# End of file
