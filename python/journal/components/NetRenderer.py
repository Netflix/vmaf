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


class NetRenderer(Component):


    def __init__(self, name="renderer"):
        Component.__init__(self, name, "net-renderer")
        self.renderer = None
        return


    def _init(self):
        from journal.devices.NetRenderer import NetRenderer
        renderer = NetRenderer()
        self.renderer = renderer
        
        return renderer



# version
__id__ = "$Id: NetRenderer.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file 
