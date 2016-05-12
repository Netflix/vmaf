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

import journal
from pyre.services.TCPService import TCPService


class JournalService(TCPService):


    class Inventory(TCPService.Inventory):

        import pyre.inventory

        marshaller = pyre.inventory.facility("marshaller", factory=journal.pickler)


    def record(self, entry):
        journal.journal().record(entry)
        return


    def generateClientConfiguration(self, registry):
        import pyre.parsing.locators
        locator = pyre.parsing.locators.simple('service')

        # get the inheriter settings
        TCPService.generateClientConfiguration(self, registry)
 
        # record the marshaller key
        # FIXME: generalize this to other picklers, like idd and ipa
        self.marshaller.generateClientConfiguration(registry)

        return


    def __init__(self, name=None):
        if name is None:
            name = 'journald'

        TCPService.__init__(self, name)

        # the remote request marshaller
        self.marshaller = None
        self._counter = 0

        return


    def _configure(self):
        TCPService._configure(self)
        self.marshaller = self.inventory.marshaller
        return


# version
__id__ = "$Id: JournalService.py,v 1.1.1.1 2006-11-27 00:09:36 aivazis Exp $"

# End of file 
