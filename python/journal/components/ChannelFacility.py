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

from pyre.inventory.Facility import Facility


class ChannelFacility(Facility):


    def __init__(self, name):
        from Channel import Channel
        Facility.__init__(self, name=name, factory=Channel, args=[name])

        return


    def _retrieveComponent(self, instance, componentName):
        from Channel import Channel
        channel = Channel(componentName)

        import pyre.parsing.locators
        locator = pyre.parsing.locators.simple('built-in')

        return channel, locator
    

# version
__id__ = "$Id: ChannelFacility.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file 
