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


class Service(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory
        from pyre.units.time import minute

        port = pyre.inventory.int('port', default=50000)
        timeout = pyre.inventory.dimensional("timeout", default=10.0*minute)


    def serve(self):
        self._info.line("%s(%s).serve" % (self.name, self.__class__.__name__))
        self._info.line("    port: %s" % self.port)
        self._info.line("    timeout: %s seconds" % self.timeout)

	# enter the event loop
        self._info.log("    entering selector watch...")
        self._serve()
        return


    def generateClientConfiguration(self, registry):
        import socket
        import pyre.parsing.locators

        locator = pyre.parsing.locators.simple('service')

        registry.setProperty('port', self.port, locator)
        registry.setProperty('host', socket.gethostbyname(socket.gethostname()), locator)

        return


    def onConnectionAttempt(self, selector, monitor):
        raise NotImplementedError(
            "class %r must override 'onConnectionAttempt'" % self.__class__.__name__)


    def onTimeout(self, selector):
        import time
        self._debug.log("MARK: %s" % time.ctime())
        return True


    def onInterrupt(self, selector):
        return


    def onReload(self, *unused):
        """event handler to reread service configuration: SIGHUP on Unix"""
        return


    def registerSignalHandlers(self):
        import signal
        signal.signal(signal.SIGHUP, self.onReload)
        return


    def validateConnection(self, address):
        return True


    def __init__(self, name, facility=None):
        if facility is None:
            facility = "service"
            
        Component.__init__(self, name, facility)

        # number of idle seconds before onTimeout gets called
        self.timeout = 0

        # the event loop and dispatcher
        self.selector = None

        # the monitor of my servcie port that gets me network connections
        self.monitor = None

        # the object repsonsible for extracting service requests from the network stream
        self.marshaller = None
        
        # the object repsonsible for translating service requests into method calls
        self.evaluator = None
        
        # someone must supply a weaver for rendering my current state in _storeConfiguration()
        # this is typically done by my parent
        self.weaver = None

        return


    def _configure(self):
        Component._configure(self)
        self.timeout = self.inventory.timeout.value
        return


    def _init(self):
        Component._init(self)

        # install the ipc support
        import pyre.ipc
        self.selector = self._createSelector()
        self.monitor = self._createPortMonitor()

        # create the command dispatcher
        self.evaluator = self._createEvaluator()

        # initialize the port monitor
        self.monitor.install(self.inventory.port)

        # register our callbacks
        self.selector.notifyWhenIdle(self.onTimeout)
        self.selector.notifyOnInterrupt(self.onInterrupt)
        self.selector.notifyOnReadReady(self.monitor, self.onConnectionAttempt)

	# register the signal handlers
        self.registerSignalHandlers()

        return


    def _serve(self):
        self.selector.watch(self.timeout)
        return


    def _createEvaluator(self):
        import pyre.services
        return pyre.services.evaluator()


    def _createSelector(self):
        import pyre.ipc
        return pyre.ipc.selector()


    def _createPortMonitor(self):
        raise NotImplementedError(
            "class %r must override '_createPortMonitor'" % self.__class__.__name__)


    def _getPort(self):
        return self.monitor.port

    
    port = property(_getPort, None, None, "my port number")
    

# version
__id__ = "$Id: Service.py,v 1.1.1.1 2006-11-27 00:10:06 aivazis Exp $"

# End of file 
