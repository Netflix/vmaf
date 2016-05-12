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


class Device(Component):


    class Inventory(Component.Inventory):

        from RendererFacility import RendererFacility

        renderer = RendererFacility()
        renderer.meta['tip'] = 'the facility that controls how the messages are formatted'


    def createDevice(self):
        raise NotImplementedError("class '%s' must override 'device'" % self.__class__.__name__)


    def __init__(self, name):
        Component.__init__(self, name, "journal-device")
        self.device = None
        return


    def _init(self):
        device = self.createDevice()
        renderer = self.inventory.renderer.renderer
        device.renderer = renderer

        self.device = device
        
        return


# version
__id__ = "$Id: Device.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file 
