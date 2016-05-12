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


class Block(AbstractNode):

    tag = "block"


    def notify(self, parent):
        block = pyre.geometry.solids.block(diagonal=self._diagonal)
        parent.onBlock(block)

        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, attributes)
        self._diagonal = self._parse(attributes["diagonal"])
        return


# version
__id__ = "$Id: Block.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
