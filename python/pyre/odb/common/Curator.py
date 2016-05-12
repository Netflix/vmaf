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


class Curator(object):


    def registerCodecs(self, *codecs):
        for codec in codecs:
            self.codecs[codec.encoding] = codec

        return


    def getShelf(self, address, encoding):
        raise NotImplementedError("class '%s' must override 'shelf'" % self.__class__.__name__)


    def __init__(self, name):
        self.codecs = {}

        tag = name + '.curator'

        import journal
        self._info = journal.info(tag)
        self._debug = journal.debug(tag)
        
        return


    class NotFoundError(Exception):


        def __init__(self, category, item):
            self.item = item
            self.category = category
            return


        def __str__(self):
            return "%s '%s' not found" % (self.category, self.item)
    

# version
__id__ = "$Id: Curator.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
