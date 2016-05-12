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


# adapted from GvR's Singleton implementation

class Singleton(object):


    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it

        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it


    def init(self, *args, **kwds):
        """constructor substitute for initializing the singleton"""
        return


# version
__id__ = "$Id: Singleton.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
