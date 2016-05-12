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


from Column import Column


class Schemer(type):


    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)

        writeable = []
        columnRegistry = {}

        # register inherited columns
        bases = list(bases)
        bases.reverse()
        for base in bases:
            try:
                columnRegistry.update(base._columnRegistry)
            except AttributeError:
                pass

            try:
                writeable += base._writeable
            except AttributeError:
                pass

        # scan the class record for columns
        for name, item in cls.__dict__.iteritems():

            # disregard entries that do not derive from Column
            if not isinstance(item, Column):
                continue

            # register it
            columnRegistry[item.name] = item
            if not item.auto:
                writeable.append(item.name)

        # install the registries into the class record
        cls._writeable = writeable
        cls._columnRegistry = columnRegistry

        return


# version
__id__ = "$Id: Schemer.py,v 1.1.1.1 2006-11-27 00:09:55 aivazis Exp $"

# End of file 
