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


from SimpleRenderer import SimpleRenderer


class TreeRenderer(SimpleRenderer):


    def onDirectory(self, node):
        self._render(node, "d")

        children = node.children()
        if not children:
            return

        # save graphics
        filler = self._filler
        graphic = self._graphic

        self._filler = filler + " | "
        self._graphic = filler + " +-"

        for entry in children[:-1]:
            entry.identify(self)

        entry = children[-1]
        self._graphic = filler + " `-"
        self._filler = filler + "   "
        entry.identify(self)

        # restore graphics
        self._filler = filler
        self._graphic = graphic

        return


    def __init__(self):
        self._filler = ""
        self._graphic = ""
        return


    def _render(self, node, code):
        print "%s (%s) (%s)" % (self._graphic, code, node.name)
        return


# version
__id__ = "$Id: TreeRenderer.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

#  End of file 
