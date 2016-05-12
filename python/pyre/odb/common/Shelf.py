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


class Shelf(dict):


    def __init__(self, name, const, codec):
        dict.__init__(self)

        self.name = name

        self._codec = codec
        self._const = const
        self._dirty = False
        self._frozen = False

        return


    def __setitem__(self, key, value):
        if self._const or self._frozen:
            raise self.AccessError(self.name, "permission denied")

        dict.__setitem__(self, key, value)
        self._dirty = True
        return


    class AccessError(Exception):


        def __init__(self, shelf, error):
            self.shelf = shelf
            self.error = error
            return


        def __str__(self):
            return "'%s': %s" % (self.shelf, self.error)


# version
__id__ = "$Id: Shelf.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
