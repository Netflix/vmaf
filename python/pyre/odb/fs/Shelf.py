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


from pyre.odb.common.Shelf import Shelf as Base


class Shelf(Base):


    def get(self, item, default=None, args=None):
        if args is None:
            args = ()

        factory = dict.get(self, item, default)
        if factory is default:
            return factory

        return factory(*args)


# version
__id__ = "$Id: Shelf.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
