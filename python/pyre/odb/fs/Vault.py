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


class Vault(object):


    def resolve(self, address):
        import os
        item = os.path.join(self.path, *address)
        return item


    def retrieveVaults(self):
        self._rep.expand()
        return self._rep.subdirectories()


    def retrieveShelves(self, address=None, extension=None):

        if address is None:
            rep = self._rep
        else:
            import os
            import pyre.filesystem
            directory = os.path.join(self._rep.path, *address)

            if not os.path.isdir(directory):
                return []
            
            rep = pyre.filesystem.root(directory)
        
        rep.expand()

        if not extension:
            return rep.files()
        
        import os
        files = []

        suffix = '.' + extension
        for node in rep.files():
            shelf, ext = os.path.splitext(node.name)

            if shelf == "__vault__":
                continue
            if ext == suffix:
                files.append(shelf)
            
        return files


    def __init__(self, rep):
        self._rep = rep
        return


    # property: name
    def _getName(self):
        return self._rep.name

    name = property(_getName, None, None, "")


    # property: path
    def _getPath(self):
        return self._rep.path

    path = property(_getPath, None, None, "")


# version
__id__ = "$Id: Vault.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
