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


class State(object):


    def get(self):
        return self._state


    def set(self, value):
        self._state = value
        return


    def activate(self):
        self._state = True
        return


    def deactivate(self):
        self._state = False
        return


    def flip(self):
        self._state ^= True
        return


    def __init__(self, initialValue):
        self._state = initialValue
        return


# version
__id__ = "$Id: State.py,v 1.1.1.1 2006-11-27 00:09:36 aivazis Exp $"

#  End of file 
