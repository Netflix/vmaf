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


from pyre.odb.common.Codec import Codec


class CodecDBM(Codec):


    def open(self, db, mode='r'):

        filename = db + '.' + self.extension

        import anydbm
        return anydbm.open(filename, mode)


    def __init__(self):
        Codec.__init__(self, encoding='dbm', extension='dbm')
        return


# version
__id__ = "$Id: CodecDBM.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
