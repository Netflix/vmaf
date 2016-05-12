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

class Visitor(object):


    # solids bodies
    def onBlock(self, block):
        return self._abstract("onBlock")


    def onCone(self, cone):
        return self._abstract("onCone")


    def onCylinder(self, cylinder):
        return self._abstract("onCylinder")


    def onPrism(self, prism):
        return self._abstract("onPrism")


    def onPyramid(self, pyramid):
        return self._abstract("onPyramid")


    def onSphere(self, sphere):
        return self._abstract("onSphere")


    def onTorus(self, torus):
        return self._abstract("onTorus")


    def onGeneralizedCone(self, cone):
        return self._abstract("onGeneralizedCone")


    # Euler operations
    def onDifference(self, difference):
        return self._abstract("onDifference")


    def onIntersection(self, intersection):
        return self._abstract("onIntersection")


    def onUnion(self, union):
        return self._abstract("onUnion")


    # transformations
    def onDilation(self, dilation):
        return self._abstract("onDilation")


    def onReflection(self, reflection):
        return self._abstract("onReflection")


    def onReversal(self, reversal):
        return self._abstract("onReversal")


    def onRotation(self, rotation):
        return self._abstract("onRotation")


    def onTranslation(self, translation):
        return self._abstract("onTranslation")



    # throw an exception
    def _abstract(self, method):
        raise NotImplementedError(
            "class '%s' should override method '%s'" % (self.__class__.__name__, method))


# version
__id__ = "$Id: Visitor.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

#
# End of file
