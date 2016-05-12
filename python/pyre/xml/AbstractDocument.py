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


class AbstractDocument(object):


    # abstract methods
    # parser.startElement handler
    def node(self, tag, attributes):
        raise NotImplementedError(
            "class '%s' should override method 'node'" % self.__class__.__name__)


    def __init__(self, source):
        # the name of the source document
        self.source = source
        
        # this attribute is meant to be set by the handler of the single document element
        self.document = None

        # this attribute is set by the parser 
        self._locator = None

        return


    # support for locator information
    def _getLocator(self):
        filename = self.source
        line = self._locator.getLineNumber()
        column = self._locator.getColumnNumber()

        import pyre.parsing.locators
        return pyre.parsing.locators.file(filename, line, column)


    def _setLocator(self, locator):
        self._locator = locator
        return
    

    locator = property(_getLocator, _setLocator, None, "")


# version
__id__ = "$Id: AbstractDocument.py,v 1.1.1.1 2006-11-27 00:10:09 aivazis Exp $"

# End of file 
