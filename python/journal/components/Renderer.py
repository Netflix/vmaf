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

from pyre.components.Component import Component


class Renderer(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory

        header = pyre.inventory.str(
            "header",
            default=" >> %(filename)s:%(line)s:%(function)s\n -- %(facility)s(%(severity)s)")
        header.meta['tip'] = "the first line of the generated message"

        footer = pyre.inventory.str("footer", default="")
        footer.meta['tip'] = "the last line of the generated message"

        format = pyre.inventory.str("format", default=" -- %s")
        format.meta['tip'] = "the format string used to render the message"


    def __init__(self, name="renderer"):
        Component.__init__(self, name, "renderer")
        self.renderer = None
        return


    def _init(self):
        from journal.devices.Renderer import Renderer
        renderer = Renderer()

        renderer.header = self.inventory.header
        renderer.footer = self.inventory.footer
        renderer.format = self.inventory.format

        self.renderer = renderer
        
        return renderer


# version
__id__ = "$Id: Renderer.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file 
