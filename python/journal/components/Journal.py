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


class Journal(Component):


    class Inventory(Component.Inventory):


        from ChannelFacility import ChannelFacility
        from DeviceFacility import DeviceFacility

        error = ChannelFacility("error")
        error.meta['tip'] = 'controls wchich error messages get printed'
        
        warning = ChannelFacility("warning")
        warning.meta['tip'] = 'controls which warning get printed'
        
        info = ChannelFacility("info")
        info.meta['tip'] = 'controls which informational messages get printed'

        debug = ChannelFacility("debug")
        debug.meta['tip'] = 'controls which debugging messages get printed'

        firewall = ChannelFacility("firewall")
        debug.meta['tip'] = 'controls which firewalls are checked'

        device = DeviceFacility()
        device.meta['tip'] = 'controls the output device used for printing the generated messages'


    def device(self):
        return self.inventory.device


    def __init__(self, name=None):
        if name is None:
            name = 'journal'
            
        Component.__init__(self, name, facility="journal")
        return


    def _init(self):
        import journal
        theJournal = journal.journal()

        device = self.inventory.device.device
        theJournal.device = device

        Component._init(self)

        return

# version
__id__ = "$Id: Journal.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file 
