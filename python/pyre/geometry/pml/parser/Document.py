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

from pyre.xml.Document import Document as DocumentNode


class Document(DocumentNode):


    tags = [
        "Geometry",
        "Block", "Cone", "Cylinder", "Prism", "Pyramid", "Sphere", "Torus", "GeneralizedCone",
        "Difference", "Intersection", "Union",
        "Dilation", "Reflection", "Reversal", "Rotation", "Translation",
        "Angle", "Scale", "Vector"
        ]
        

    def onGeometry(self, bodies):
        self.document = bodies
        return


# version
__id__ = "$Id: Document.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
