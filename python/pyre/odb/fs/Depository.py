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


from Vault import Vault


class Depository(Vault):


    def createDepository(self, *address):
        import os
        directory = os.path.join(self._rep.path, *address)
        if os.path.isdir(directory):
            return Depository(directory)

        return None


    def __init__(self, directory):
        import pyre.filesystem
        rep = pyre.filesystem.root(directory)

        Vault.__init__(self, rep)

        return

# version
__id__ = "$Id: Depository.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
