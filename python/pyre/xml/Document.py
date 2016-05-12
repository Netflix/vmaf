#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from AbstractDocument import AbstractDocument


class Document(AbstractDocument):


    tags = []


    # the metaclass has prepared a look up table of nested tags
    def node(self, tag, attributes):
        return self._mydtd[tag](self, attributes)


    def __init__(self, source):
        AbstractDocument.__init__(self, source)
        return


    # build the lookup table
    from DTDBuilder import DTDBuilder
    __metaclass__ = DTDBuilder
    del DTDBuilder


# version
__id__ = "$Id: Document.py,v 1.1.1.1 2006-11-27 00:10:10 aivazis Exp $"

# End of file 
