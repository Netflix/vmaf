#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


class Selector(object):


    def watch(self, timeout=None):
        """dispatch events to the registered hanlders"""
        
        if timeout:
            self._timeout = timeout

        self._watch()
        return
    
        # FIXME:
        # leave like this until I understand better the set of exceptions I
        # would like to handle. It really is bad to catch all exceptions,
        # especially since it hides errors during development
        try:
            self._watch()

        except:
            # catch all exceptions
            self._cleanup()

            # get exception information
            import sys
            type, value = sys.exc_info()[:2]

            # rethrow the exception so the clients can handle it
            raise type, value

        return


    def notifyOnReadReady(self, fd, handler):
        """add <handler> to the list of routines to call when <fd> is read ready"""
        self._input.setdefault(fd, []).append(handler)
        return


    def notifyOnWriteReady(self, fd, handler):
        """add <handler> to the list of routines to call when <fd> is write ready"""
        self._output.setdefault(fd, []).append(handler)
        return


    def notifyOnException(self, fd, handler):
        """add <handler> to the list of routines to call when <fd> raises an exception"""
        self._exception.setdefault(fd, []).append(handler)
        return


    def notifyOnInterrupt(self, handler):
        """add <handler> to the list of routines to call when a signal arrives"""
        self._interrupt.append(handler)
        return


    def notifyWhenIdle(self, handler):
        """add <handler> to the list of routines to call when a timeout occurs"""
        self._idle.append(handler)
        return


    def __init__(self):
        self.state = True
        self._timeout = self._TIMEOUT

        # the fd activity clients
        self._input = {}
        self._output = {}
        self._exception = {}

        # clients to notify when there is nothing else to do
        self._idle = []
        self._interrupt = []
        
        return


    def _watch(self):
        import select

        while self.state:

            self._debug.line("constructing list of watchers")
            iwtd = self._input.keys()
            owtd = self._output.keys()
            ewtd = self._exception.keys()

            self._debug.line("input: %s" % iwtd)
            self._debug.line("output: %s" % owtd)
            self._debug.line("exception: %s" % ewtd)

            self._debug.line("checking for indefinite block")
            if not iwtd and not owtd and not ewtd and not self._idle:
                self._debug.log("no registered handlers left; exiting")
                return

            self._debug.line("calling select")
            try:
                reads, writes, excepts = select.select(iwtd, owtd, ewtd, self._timeout)
            except select.error, error:
                # GUESS:
                # when a signal is delivered to a signal handler registered
                # by the application, the select call is interrupted and
                # raises a select.error
                errno, msg = error
                self._debug.log("signal received: %d: %s" % (errno, msg))
                continue
                
            self._debug.line("returned from select")

            # dispatch to the idle handlers if this was a timeout
            if not reads and not writes and not excepts:
                self._debug.log("no activity; dispatching to idle handlers")
                for handler in self._idle:
                    if not handler(self):
                        self._idle.remove(handler)
            else:
                # dispatch to the registered handlers
                self._debug.log("dispatching to exception handlers")
                self._dispatch(self._exception, excepts)
                self._debug.log("dispatching to output handlers")
                self._dispatch(self._output, writes)
                self._debug.log("dispatching to input handlers")
                self._dispatch(self._input, reads)

        return


    def _dispatch(self, handlers, entities):

        for fd in entities:
            for handler in handlers[fd]:
                if not handler(self, fd):
                    handlers[fd].remove(handler)
            if not handlers[fd]:
                del handlers[fd]

        return


    def _cleanup(self):
        self._debug.log("cleaning up")
        for fd in self._input:
            fd.close()
        for fd in self._output:
            fd.close()
        for fd in self._exception:
            fd.close()

        for handler in self._interrupt:
            handler(self)

        return
        

    # static members
    import journal
    _debug = journal.debug("pyre.ipc.selector")
    del journal


    # constants
    _TIMEOUT = .5


# version
__id__ = "$Id: Selector.py,v 1.1.1.1 2006-11-27 00:10:04 aivazis Exp $"

#  End of file 
