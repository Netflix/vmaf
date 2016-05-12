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


from Mill import Mill
from LineComments import LineComments

class LineMill(Mill, LineComments):


    def __init__(self, comment, firstLine):
        Mill.__init__(self)
        LineComments.__init__(self)

        self.commentLine = comment
        self.firstLine = firstLine

        return


# version
__id__ = "$Id: LineMill.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
