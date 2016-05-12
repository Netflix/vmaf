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


from Device import Device


class Remote(Device):


    class Inventory(Device.Inventory):

        import pyre.inventory
        from NetRenderer import NetRenderer
        from RendererFacility import RendererFacility

        key = pyre.inventory.str("key")
        key.meta['tip'] = (
            "the passkey of the remote service -- normally, this is set automatically")
        host = pyre.inventory.str("host", default="localhost")

        host.meta['tip'] = "the hostname where the remote journal service is running"
        
        port = pyre.inventory.int("port", default=50000)
        port.validator = pyre.inventory.range(1024+1, 64*1024-1)
        port.meta['tip'] = (
            "the port that the remote journal service is monitoring for incoming requests")

        renderer = RendererFacility(factory=NetRenderer)
        renderer.meta['tip'] = "the facility that controls how the messages are formatted"


    def createDevice(self):

        key = self.inventory.key
        host = self.inventory.host
        port = self.inventory.port

        import journal
        return journal.remote(key=key, port=port, host=host)


    def __init__(self):
        Device.__init__(self, "remote")
        return


# version
__id__ = "$Id: Remote.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file 
