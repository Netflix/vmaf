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


class Codec(object):


    def open(self, db, mode):
        raise NotImplementedError("class '%s' must override 'open'" % self.__class__.__name__)


    def __init__(self, encoding, extension):
        self.encoding = encoding
        self.extension = extension
        return


# version
__id__ = "$Id: Codec.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
