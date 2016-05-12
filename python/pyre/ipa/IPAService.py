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


from pyre.services.TCPService import TCPService


class IPAService(TCPService):


    class Inventory(TCPService.Inventory):

        import pyre.ipa
        import pyre.inventory
        from pyre.units.time import hour

        ticketOnce = pyre.inventory.bool("ticketOnce", default=True)
        ticketDuration = pyre.inventory.dimensional("ticketDuration", default=0.5*hour)

        marshaller = pyre.inventory.facility("marshaller", factory=pyre.ipa.pickler)
        userManager = pyre.inventory.facility("userManager", factory=pyre.ipa.userManager)


    def generateClientConfiguration(self, registry):
        """update the given registry node with sufficient information to grant access to clients"""

        import pyre.parsing.locators
        locator = pyre.parsing.locators.simple('service')

        # get the inherited settings
        TCPService.generateClientConfiguration(self, registry)

        # record the marshaller name
        registry.setProperty('marshaller', self.marshaller.name, locator)

        # get the marshaller to record his configuration
        marshaller = registry.getNode(self.marshaller.name)
        self.marshaller.generateClientConfiguration(marshaller)

        return


    def login(self, username, cleartext):
        password = self.userManager.authenticate(username, cleartext)

        if not password:
            self._info.log("rejected password %r from user %r" % (cleartext, username))
            return
        
        ticket = self._createTicket(username, password)
        self._info.log("issued ticket %r to user %r" % (ticket, username))
        return ticket


    def refresh(self, username, ticket):
        if ticket not in self._tickets:
            self._info.log("got bad ticket %r" % ticket)
            return

        if ticket[:len(username)] != username:
            self._info.log("username does not match ticket %r" % ticket)
            return

        self._info.log("got good ticket %r" % ticket)

        expiration = self._tickets[ticket]

        # one time tickets are invalidated immediately -- the rest just expire
        # this allows the browser's back key to function for a while
        ticketOnce = self.inventory.ticketOnce
        if ticketOnce:
            del self._tickets[ticket]

        import time
        if time.time() > expiration:
            self._info.log("got expired ticket %r" % (ticket))
            return

        newTicket = self._createTicket(username, ticket[len(username):])
        self._info.log("exchanged good ticket %r for %r" % (ticket, newTicket))
        return newTicket


    def verify(self, username, ticket):
        if ticket not in self._tickets:
            self._info.log("got bad ticket %r" % ticket)
            return False

        if ticket[:len(username)] != username:
            self._info.log("username does not match ticket %r" % ticket)
            return False

        self._info.log("got good ticket %r" % ticket)

        return True


    def logout(self, username, ticket):
        if ticket not in self._tickets:
            self._info.log("got bad ticket %r" % ticket)
            return False

        if ticket[:len(username)] != username:
            self._info.log("username does not match ticket %r" % ticket)
            return False

        self._info.log("got good ticket %r" % ticket)
        
        del self._tickets[ticket]
        self._info.log("deleted good ticket %r" % ticket)

        return True


    def onTimeout(self, selector):
        import time
        self._info.log("thump")

        now = time.time()

        expired = []
        for ticket, timestamp in self._tickets.iteritems():
            self._info.log("ticket %r: %s seconds until expiration" % (ticket, timestamp - now))
            if now > timestamp:
                expired.append(ticket)

        for ticket in expired:
            self._info.log("removed stale ticket %r issued on %r" % (ticket, timestamp))
            del self._tickets[ticket]

        self._info.log("unexpired tickets: %s" % len(self._tickets))

        return True


    def onReload(self, *unused):
        return self.userManager.onReload()


    def __init__(self, name=None):
        if name is None:
            name = "ipa"
            
        TCPService.__init__(self, name)

        # the user manager
        self.userManager = None

        # private data
        self._tickets = {}

        return


    def _configure(self):
        TCPService._configure(self)
        self.marshaller = self.inventory.marshaller
        self.userManager = self.inventory.userManager
        return


    def _createTicket(self, seed, text):
        import time
        import random
        
        t = list(text)
        random.shuffle(t)
        shuffled = "".join(t)

        ticket = seed + shuffled
        self._tickets[ticket] = time.time() + self.inventory.ticketDuration.value

        return ticket


# version
__id__ = "$Id: IPAService.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

# End of file 
