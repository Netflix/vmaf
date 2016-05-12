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


import journal._journal as proxy


class ProxyState(object):


    def get(self):
        return proxy.getState(self._handle)


    def set(self, value):
        return proxy.setState(self._handle, value)


    def activate(self):
        proxy.activate(self._handle)
        return


    def deactivate(self):
        proxy.deactivate(self._handle)
        return


    def flip(self):
        proxy.flip(self._handle)
        return


    def __init__(self, handle):
        self._handle = handle
        return


# version
__id__ = "$Id: ProxyState.py,v 1.1.1.1 2006-11-27 00:09:36 aivazis Exp $"

#  End of file 
