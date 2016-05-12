#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2007 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


class Locator(object):


    def __init__(self, file, line, column):

        self.source = file
        self.line = line
        self.column = column

        return


    __slots__ = ("column", "source", "line")


# version
__id__ = "$Id: Locator.py,v 1.2 2007-09-13 17:12:17 aivazis Exp $"

#  End of file 
