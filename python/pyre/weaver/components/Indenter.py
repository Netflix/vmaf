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


class Indenter(object):


    def __init__(self, indentMark=None):
        self._margin = ""
        self._indentationLevel = 0
        if indentMark is None:
            self._indentMark = self._INDENT_MARK
        else:
            self._indentMark = indentMark
        return


    def _indent(self):
        self._indentationLevel += 1
        self._margin = self._indentMark * self._indentationLevel
        return


    def _outdent(self):
        self._indentationLevel -= 1
        self._margin = self._indentMark * self._indentationLevel
        return


    def _leader(self):
        return self._margin


    def _render(self, text):
        return self._margin + text


    _INDENT_MARK = "    "


# version
__id__ = "$Id: Indenter.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
