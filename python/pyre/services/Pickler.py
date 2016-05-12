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


import pickle
from Marshaller import Marshaller


def createKey():
    alphabet = list("0123456789abcdefghijklmnopqrstuvwxyz")

    import random
    random.shuffle(alphabet)
    key = "".join(alphabet)[0:16]

    return key
        

class Pickler(Marshaller):


    class Inventory(Marshaller.Inventory):

        import pyre.inventory

        key = pyre.inventory.str("key", default=createKey())
        

    def send(self, data, socket):
        stream = socket.makefile("wb", 0)

        self._debug.log("sending data: %s" % data)
        request = Request(self.key, data)
    
        try:
            pickle.dump(request, stream)
        except EOFError:
            text = '%s: unable to send request: EOFError' % self.__class__.__name__
            raise self.RequestError(text)
        except IOError, msg:
            text = '%s: unable to send request: IOError: %s' % (self.__class.__name, msg)
            raise self.RequestError(text)

        return


    def receive(self, socket):
        stream = socket.makefile("rb")

        try:
            request = pickle.load(stream)
        except EOFError:
            text = '%s: unable to receive request: EOFError' % self.__class__.__name__
            raise self.RequestError(text)
        except IOError, msg:
            text = '%s: unable to receive request: IOError: %s' % (self.__class.__name, msg)
            raise self.RequestError(text)

        self._debug.log("received request: key=%s, data=%s" % (request.key, request.data))

        return self.authenticate(request).data


    def generateClientConfiguration(self, registry):
        import pyre.parsing.locators
        locator = pyre.parsing.locators.simple('service')
        registry.setProperty('key', self.key, locator)
        return


    def authenticate(self, request):
        if request.key == self.key:
            self._debug.log("accepted key {%s}" % request.key)
            return request
        
        raise ValueError, "%s: key mismatch: %r(mine) != %r(client's)" % (
            self.__class__.__name__, self.key, request.key)


    def __init__(self, name=None):
        if name is None:
            name = "pickler"

        Marshaller.__init__(self, name)

        self.key = ''

        return


    def _configure(self):
        Marshaller._configure(self)
        self.key = self.inventory.key
        return


class Request(object):


    def __init__(self, key, data):
        self.key = key
        self.data = data
        return


# version
__id__ = "$Id: Pickler.py,v 1.1.1.1 2006-11-27 00:10:06 aivazis Exp $"

# End of file 
