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


from pyre.odb.common.Curator import Curator as Base


class Curator(Base):


    # shelves
    def retrieveShelves(self, address, extension, extraDepositories=[]):
        shelves = []
        for depository in self.searchOrder(extraDepositories):
            shelves += depository.retrieveShelves(address, extension)

        return shelves


    def retrieveShelfNames(self, address, extension, extraDepositories=[]):
        files = []
        for depository in self.searchOrder(extraDepositories):
            candidates = depository.retrieveShelves(address, extension)
            files += [
                depository.resolve(address + [name]) + '.' + extension for name in candidates ]

        return files


    def resolve(self, address, extraDepositories=[]):
        import os
        
        names = []
        for depository in self.searchOrder(extraDepositories):
            path = depository.resolve(address)
            if os.path.exists(path):
                names.append(path)

        return names


    def loadSymbol(self, tag, codec, address, symbol, extras=[], errorHandler=None):
        """extract <symbol> from a shelf pointed to by <address>"""

        import pyre.parsing.locators
        if not tag:
            tag = symbol

        # loop over the depositories
        for depository in self.searchOrder(extraDepositories=extras):
            spec = depository.resolve(address)
            filename = codec.resolve(spec)

            locator = pyre.parsing.locators.file(filename)

            # open the shelf
            try:
                shelf = codec.open(spec, 'r')
            except IOError, error:
                # the codec failed to open the spec
                if callable(errorHandler):
                    errorHandler(tag, locator, error)
                continue

            # retrieve the factory method
            try:
                item = shelf[symbol]
            except KeyError:
                # no factory by that name exists
                if callable(errorHandler):
                    errorHandler(tag, locator, "'%s' not found" % symbol)
                continue

            # success
            yield item, locator
            
        return


    # depository management
    def createDepository(self, directory):
        """create a new depository rooted at <directory>"""

        import os
        if os.path.isdir(directory):
            from Depository import Depository
            depository = Depository(directory)
            return depository

        return None


    def addDepositories(self, *directories):
        """create new depositories out of <directories> and add them to the search list"""
        return [ self.addDepository(directory) for directory in directories ]


    def addDepository(self, directory):
        """create a new depository and add it to the search list"""

        depository = self.createDepository(directory)
        if depository:
            self.depositories.append(depository)
            
        return depository


    # searches through the list of depositories
    def searchOrder(self, extraDepositories=[]):
        """walk through my depositories in-order, resolving address"""

        return self.depositories + extraDepositories


    def __init__(self, name):
        Base.__init__(self, name)
        self.depositories = []
        return


    def _loadObject(self, codec, spec, name, args):
        """attempt to load object created by factory method <name> in <spec>"""

        shelf = codec.open(spec, 'r')
        factory = shelf[name]
        item = factory(*args)
        
        return item


# version
__id__ = "$Id: Curator.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
