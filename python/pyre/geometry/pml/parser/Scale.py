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

from AbstractNode import AbstractNode


class Scale(AbstractNode):

    tag = "scale"


    def content(self, content):
        self._scale += content
        return


    def notify(self, parent):
        value = float(self._scale.strip())
        parent.onScale(value)
        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, attributes)
        self._scale = ''
        return


# version
__id__ = "$Id: Scale.py,v 1.1.1.1 2006-11-27 00:09:58 aivazis Exp $"

# End of file
