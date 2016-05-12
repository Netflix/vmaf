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


from Directory import Directory


class Root(Directory):


    def __init__(self, name):
        Directory.__init__(self, name, None)
        return


    def _buildPath(self, parts):
        parts.append(self.name)
        return



# version
__id__ = "$Id: Root.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

#  End of file 
