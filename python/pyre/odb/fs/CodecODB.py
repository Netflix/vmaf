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


class CodecODB(Codec):


    def open(self, db, mode='r'):
        """open the file <db> in mode <mode> and place its contents in a shelf"""

        filename = self.resolve(db)

        import os
        exists = os.path.isfile(filename)
        
        if mode in ['w'] and not exists:
            raise IOError("file not found: '%s'" % filename)

        shelf = self._shelf(filename, False)
        self._decode(shelf)

        if mode == 'r':
            shelf._const = True
        else:
            shelf._const = False

        return shelf


    def resolve(self, db):
        return db + '.' + self.extension


    def __init__(self, encoding, extension=None):
        if extension is None:
            extension = encoding
            
        Codec.__init__(self, encoding, extension)

        # public data
        self.renderer = self._createRenderer()

        # private data
        self._locker = self._createLocker()
        
        return


    def _shelf(self, filename, const):
        """create a shelf for the contents of the db file"""

        from Shelf import Shelf
        return Shelf(filename, const, self)


    def _decode(self, shelf):
        """lock and then read the contents of the file into the shelf"""

        stream = file(shelf.name)

        self._locker.lock(stream, self._locker.LOCK_EX)
        exec stream in shelf
        self._locker.unlock(stream)

        return


    def _createRenderer(self):
        """create a weaver for storing shelves"""
        
        from pyre.weaver.Weaver import Weaver
        weaver = Weaver()
        return weaver


    def _createLocker(self):
        from FileLocking import FileLocking
        return FileLocking()

        
# version
__id__ = "$Id: CodecODB.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
