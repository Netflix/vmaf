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

class PerlMill(LineMill):


    names = ["perl"]


    def __init__(self):
        LineMill.__init__(self, "#", "#!/usr/bin/env perl")
        return


# version
__id__ = "$Id: PerlMill.py,v 1.1.1.1 2006-11-27 00:10:09 aivazis Exp $"

#  End of file 
