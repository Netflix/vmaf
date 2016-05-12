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
from BlockComments import BlockComments

class BlockMill(Mill, BlockComments):


    def __init__(self, begin, line, end, firstline):
        Mill.__init__(self)
        BlockComments.__init__(self)

        self.commentBlockLine = line
        self.commentBeginBlock = begin
        self.commentEndBlock = end
        self.firstLine = firstline

        return


# version
__id__ = "$Id: BlockMill.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
