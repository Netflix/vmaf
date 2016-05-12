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


class Interface(type):


    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)

        import types

        interfaceRegistry = {}
        
        for name, record in dict.iteritems():
            if name[0] == '_':
                continue
            
            if not isinstance(record, types.FunctionType):
                continue

            print name
            interfaceRegistry[name] = None

        cls._interfaceRegistry = interfaceRegistry

        return

# version
__id__ = "$Id: Interface.py,v 1.1.1.1 2006-11-27 00:10:00 aivazis Exp $"

# End of file 
