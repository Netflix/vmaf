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

from Composition import Composition


class Geometry(Composition):

    tag = "geometry"


    def notify(self, parent):
        parent.onGeometry(self._bodies)
        return


    def __init__(self, document, attributes):
        Composition.__init__(self, attributes)
        self._bodies = []
        return


    def _setOperand(self, body):
        self._bodies.append(body)
        return


# version
__id__ = "$Id: Geometry.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
