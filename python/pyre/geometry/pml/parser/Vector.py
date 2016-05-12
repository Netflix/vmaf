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


class Vector(AbstractNode):

    tag = "vector"


    def content(self, content):
        self._vector += content
        return


    def notify(self, parent):
        vector = self._parse(self._vector.strip())
        parent.onVector(vector)
        return


    def __init__(self, document, attributes):
        AbstractNode.__init__(self, attributes)
        self._vector = ''
        return


# version
__id__ = "$Id: Vector.py,v 1.1.1.1 2006-11-27 00:09:58 aivazis Exp $"

# End of file
