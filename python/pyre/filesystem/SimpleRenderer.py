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


from Inspector import Inspector


class SimpleRenderer(Inspector):

    _INDENT = " "*4


    def onCharacterDevice(self, node):
        self._render(node, "c")
        return


    def onBlockDevice(self, node):
        self._render(node, "b")
        return


    def onDirectory(self, node):
        self._render(node, "d")
        self._indent += 1

        for entry in node.children():
            entry.identify(self)

        self._indent -= 1

        return


    def onFile(self, node):
        self._render(node, "f")
        return


    def onLink(self, node):
        self._render(node, "l")
        return


    def onNamedPipe(self, node):
        self._render(node, "p")
        return


    def onSocket(self, node):
        self._render(node, "s")
        return


    def render(self, node):
        node.identify(self)
        return


    def __init__(self):
        self._indent = 0
        return


    def _render(self, node, code):
        print "%s(%s) %s" % (self._INDENT*self._indent, code, node.name)
        return


# version
__id__ = "$Id: SimpleRenderer.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

#  End of file 
