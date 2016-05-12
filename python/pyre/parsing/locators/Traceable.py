#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class Traceable(object):


    def setLocator(self, locator):
        """set my locator to <locator>; if I already have one, chain them"""
        
        if self.locator is not None:
            import pyre.parsing.locators
            locator = pyre.parsing.locators.chain(locator, self.locator)

        self.locator = locator

        return locator


    def getLocator(self):
        """return my locator"""
        return self.locator


    def __init__(self):
        self.locator = None
        return


# version
__id__ = "$Id: Traceable.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
