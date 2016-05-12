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


class Token(object):


    def __init__(self):
        self.tid = None
        self.date = None
        self.locator = None

        return


    def __str__(self):
        return "locator=%r, id=%r, date=%r" % (self.locator, self.tid, self.date)


# version
__id__ = "$Id: Token.py,v 1.1.1.1 2006-11-27 00:10:00 aivazis Exp $"

#  End of file 
