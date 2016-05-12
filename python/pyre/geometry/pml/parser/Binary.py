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


class Binary(Composition):


    def __init__(self, document, attributes=None):
        Composition.__init__(self, attributes)
        
        self._b1 = None
        self._b2 = None

        return


    def _setOperand(self, body):
        if not self._b1:
            self._b1 = body
        elif not self._b2:
            self._b2 = body
        else:
            raise ValueError("too many nested tags in '%s'" % self.tag)
            
        return


# version
__id__ = "$Id: Binary.py,v 1.1.1.1 2006-11-27 00:09:57 aivazis Exp $"

# End of file
