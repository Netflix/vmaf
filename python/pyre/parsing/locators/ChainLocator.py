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


class ChainLocator(object):


    def __init__(self, this, next):
        self.this = this
        self.next = next
        return


    def __str__(self):
        return "%s via %s" % (self.this, self.next)


    __slots__ = ("this", "next")
    

# version
__id__ = "$Id: ChainLocator.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
