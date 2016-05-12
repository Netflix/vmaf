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


class Warning(Index):


    def init(self):
        Index.init(self, "warning", True)
        if journal.hasProxy:
            self._stateFactory = self._proxyState
        return


    def _proxyState(self, name):
        from ProxyState import ProxyState
        return ProxyState(journal._journal.warning(name))


# version
__id__ = "$Id: Warning.py,v 1.1.1.1 2006-11-27 00:09:36 aivazis Exp $"

#  End of file 
