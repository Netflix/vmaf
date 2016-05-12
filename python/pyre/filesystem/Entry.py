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


class Entry(object):


    def identify(self, inspector):
        raise NotImplementedError("class '%s' must override 'id'" % self.__class__.__name__)


    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent

        self._path = None
        
        return


    # property path
    def _getPath(self):
        if self._path is not None:
            return self._path
        
        parts = []
        self._buildPath(parts)

        import os
        self._path = os.path.join(*parts)
        return self._path


    def _buildPath(self, parts):
        self.parent._buildPath(parts)
        parts.append(self.name)
        return


    path = property(_getPath, None, None, "")


# version
__id__ = "$Id: Entry.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

#  End of file 
