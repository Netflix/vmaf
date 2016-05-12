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


import journal
from Index import Index


class Debug(Index):


    def init(self):
        Index.init(self, "debug", False)
        if journal.hasProxy:
            self._stateFactory = self._proxyState
        return


    def _proxyState(self, name):
        from ProxyState import ProxyState
        return ProxyState(journal._journal.debug(name))


# version
__id__ = "$Id: Debug.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

#  End of file 
