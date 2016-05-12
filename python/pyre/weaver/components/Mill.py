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


from Indenter import Indenter
from Stationery import Stationery
from pyre.parsing.locators.Traceable import Traceable


class Mill(Stationery, Indenter, Traceable):


    def weave(self, document=None):

        self.begin()
        if document:
            self._renderDocument(document)
        self.end()

        return self._rep


    def document(self):
        return self._rep


    def begin(self):
        self._rep = self.header()
        return


    def contents(self, body):
        self._rep += body
        return


    def end(self):
        self._versionId()
        if self._options.timestamp:
            self._rep += ['', self.line(self._timestamp())]
        self._rep += self.footer()
        return


    def __init__(self, name=None):
        if name is None:
            name = 'weaver'
            
        Stationery.__init__(self, name)
        Indenter.__init__(self)
        Traceable.__init__(self)

        self._rep = []

        return


    def _renderDocument(self, document):
        raise NotImplementedError(
            "class '%s' must override '_renderDocument'" % self.__class__.__name__)


    def _separator(self):
        self._rep.append(self.line(self.separator()))
        return


    def _versionId(self):
        format = self._options.versionId

        if format:
            self._rep += ['', self.line(" version"), self.line(format)]

        return


    def _timestamp(self):
        format = self._options.timestampLine
        creator = self._options.creator
        if not creator:
            creator = self.__class__.__name__

        if format:
            import time
            timestamp = format % (creator, time.asctime())
            return timestamp

        return ""


    def _write(self, text=''):
        if text:
            self._rep.append(self._margin + text)
        else:
            self._rep.append('')
        return


# version
__id__ = "$Id: Mill.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
