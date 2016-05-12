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

import pyre.geometry.operations
from Transformation import Transformation


class Translation(Transformation):


    tag = "translation"


    def onVector(self, vector):
        self._vector = vector
        return


    def notify(self, parent):
        if not self._body:
            raise ValueError("no body specified in '%s'" % self.tag)
        if not self._vector:
            raise ValueError("no vector specified in '%s'" % self.tag)

        translation = pyre.geometry.operations.translate(body=self._body, vector=self._vector)
        parent.onTranslation(translation)
        return


    def __init__(self, document, attributes):
        Transformation.__init__(self, attributes)
        self._vector = None
        return


# version
__id__ = "$Id: Translation.py,v 1.1.1.1 2006-11-27 00:09:58 aivazis Exp $"

# End of file
