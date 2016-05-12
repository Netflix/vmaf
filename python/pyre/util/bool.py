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

TRUE = true = on = yes = 1
FALSE = false = off = no = 0


# convert strings to bool

def bool(response):
    return _stringToBool[response.lower()]


_stringToBool = {
    "1": True,
    "y" : True,
    "yes" : True,
    "on" : True,
    "t" : True,
    "true" : True,
    "0": False,
    "n" : False,
    "no" : False,
    "off" : False,
    "f" : False,
    "false" : False,
    }
    

# version
__id__ = "$Id: bool.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
