#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from Application import Application
from Stager import Stager


class Script(Application, Stager):


    def __init__(self, name):
        Application.__init__(self, name)
        Stager.__init__(self)
        return


# version
__id__ = "$Id: Script.py,v 1.1.1.1 2006-11-27 00:09:54 aivazis Exp $"

# End of file 
