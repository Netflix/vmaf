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

# factory

def timingCenter():
    global _theTimingCenter
    if _theTimingCenter is None:
        _theTimingCenter = TimingCenter()

    return _theTimingCenter


# implementation

from pyre.util.ResourceManager import ResourceManager


class TimingCenter(ResourceManager):


    def timer(self, name):
        timer = self.find(name)
        if not timer:
            from Timer import Timer
            timer = Timer(name).activate()
            self.manage(timer, name)

        return timer


    def __init__(self):
        ResourceManager.__init__(self, "timers")
        return


# the instance

_theTimingCenter = None


# version
__id__ = "$Id: TimingCenter.py,v 1.1.1.1 2006-11-27 00:10:04 aivazis Exp $"

#  End of file 
