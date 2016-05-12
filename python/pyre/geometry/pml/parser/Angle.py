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


class Angle(AbstractNode):


    tag = "angle"


    def content(self, content):
        self._angle += content
        return


    def notify(self, parent):
        value = float(self._angle.strip())
        parent.onAngle(value)
        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, attributes)
        self._angle = ''
        return


# version
__id__ = "$Id: Angle.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
