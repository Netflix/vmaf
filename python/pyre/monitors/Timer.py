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


import time
from pyre.util.Toggle import Toggle
from pyre.util.Resource import Resource


class Timer(Resource, Toggle):


    def start(self):
        if self.state:
            self._start = time.clock()

        return self


    def stop(self):
        now = time.clock()
        self._accumulatedTime += now - self._start
        self._start = now

        return self._accumulatedTime


    def lap(self):
        if self.state:
            now = time.clock()
            return self._accumulatedTime + (now - self._start)

        return 0


    def read(self):
        return self._accumulatedTime


    def reset(self):
        self._accumulatedTime = 0
        return self


    def __init__(self, name):
        Resource.__init__(self, name)
        Toggle.__init__(self)

        self._start = time.clock()
        self._accumulatedTime = 0

        return


# version
__id__ = "$Id: Timer.py,v 1.1.1.1 2006-11-27 00:10:04 aivazis Exp $"

#  End of file 
