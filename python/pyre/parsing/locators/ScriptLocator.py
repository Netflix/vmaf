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


class ScriptLocator(object):


    def __init__(self, source, line, function):
        self.source = source
        self.line = line
        self.function = function
        return


    def __str__(self):
        return "{file=%r, line=%r, function=%r}" % (self.source, self.line, self.function)
    


    __slots__ = ("source", "line", "function")

# version
__id__ = "$Id: ScriptLocator.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
