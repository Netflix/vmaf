#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


class Journal(object):


    def record(self, entry):
        self.device.record(entry)
        return


    def entry(self):
        from diagnostics.Entry import Entry
        return Entry()


    def channel(self, name, channel=None):
        if channel is None:
            return self._channels.get(name)

        self._channels[name] = channel

        return channel


    def channels(self):
        return self._channels.keys()


    def __init__(self, name, device=None):
        self.name = name

        if device is None:
            from devices.Console import Console
            device = Console()

        self.device = device

        # private data
        self._channels = {}

        return


# version
__id__ = "$Id: Journal.py,v 1.1.1.1 2006-11-27 00:09:34 aivazis Exp $"

#  End of file 
