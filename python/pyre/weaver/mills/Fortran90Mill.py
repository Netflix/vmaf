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


from pyre.weaver.components.LineMill import LineMill

class Fortran90Mill(LineMill):


    names = ["fortran90", "fortran95", "f90", "f95"]


    def __init__(self):
        LineMill.__init__(self, "!", "! -*- F90 -*-")
        return


# version
__id__ = "$Id: Fortran90Mill.py,v 1.1.1.1 2006-11-27 00:10:09 aivazis Exp $"

#  End of file 
