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


from pyre.components.Component import Component


class Marshaller(Component):


    def send(self, request, stream):
        raise NotImplementedError("class '%s' must override 'evaluate'" % self.__class__.__name__)


    def receive(self, stream):
        raise NotImplementedError("class '%s' must override 'evaluate'" % self.__class__.__name__)


    def generateClientConfiguration(self, registry):
        raise NotImplementedError(
            "class %r must override 'generateClientConfiguration'" % self.__class__.__name)

    
    def __init__(self, name):
        Component.__init__(self, name, facility="marshaller")
        return


    from RequestError import RequestError


# version
__id__ = "$Id: Marshaller.py,v 1.1.1.1 2006-11-27 00:10:06 aivazis Exp $"

# End of file 
