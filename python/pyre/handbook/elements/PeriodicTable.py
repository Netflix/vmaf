#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""create the singleton thePeriodicTable and populate it with the known elements"""


class PeriodicTable(object):


    def name(self, name):
        return self._nameIndex.get(name)


    def symbol(self, symbol):
        return self._symbolIndex.get(symbol)


    def atomicNumber(self, n):
        if n < 1 or n > len(self._atomicNumberIndex):
            import journal
            journal.firewall("pyre.handbook").log(
                "element with atomic number %d not found" % atomicNumber)
            return None
        
        return self._atomicNumberIndex[n-1]


    def __init__(self):
        from elements import elements

        self._atomicNumberIndex = elements
        self._nameIndex = createNameIndex(elements)
        self._symbolIndex = createSymbolIndex(elements)

        return


# helpers

# verify that atomic numbers correlate well with position in list

def verify(elements):

    status = 1

    for index in range(len(elements)):
        if index + 1 != elements[index].atomicNumber:
            # when firewall are not fatal, this will scan through the table
            # and find all inconsistencies
            status = 0
            import journal
            firewall = journal.firewall("handbook")
            firewall.log(
                "PeriodicTable: atomic number(%d) != offset(%d)" %
                (index, elements[index].atomicNumber))

    return status


def createNameIndex(elements):
    index = {}

    # place all element in the index
    for element in elements:
        index[element.name] = element

    # detect collisions
    if len(elements) != len(index):
        import journal
        firewall = journal.firewall("handbook")
        firewall.log(
            "PeriodicTable: symbol index size mismatch: %d != %d" % (len(index), len(elements)))

    return index


def createSymbolIndex(elements):
    index = {}

    # place all element in the index
    for element in elements:
        index[element.symbol] = element

    # detect collisions
    if len(elements) != len(index):
        import journal
        firewall = journal.firewall("handbook")
        firewall.log(
            "PeriodicTable: symbol index size mismatch: %d != %d" % (len(index), len(elements)))

    return index


# the singleton

_thePeriodicTable = None

def periodicTable():

    import journal
    info = journal.debug("pyre.initialization")
    
    global _thePeriodicTable
    if not _thePeriodicTable:
        info.log("generating the periodic table ...")
        _thePeriodicTable = PeriodicTable()
        info.log("    done")

    return _thePeriodicTable


# version
__id__ = "$Id: PeriodicTable.py,v 1.1.1.1 2006-11-27 00:09:59 aivazis Exp $"

# End of file
