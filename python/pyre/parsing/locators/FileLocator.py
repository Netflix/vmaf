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


class FileLocator(object):


    def __init__(self, source, line, column):
        self.source = source
        self.line = line
        self.column = column
        return


    def __str__(self):
        return "{file=%r, line=%r, column=%r}" % (self.source, self.line, self.column)
    

    __slots__ = ("source", "line", "column")

# version
__id__ = "$Id: FileLocator.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
