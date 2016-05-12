#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


import journal
import traceback
from Entry import Entry


class Diagnostic(object):


    def line(self, message):
        if not self.state:
            return

        self._entry.line(message)
        return self


    def log(self, message=None):
        if not self.state:
            return

        if message is not None:
            self._entry.line(message)

        stackDepth = -2
        stackTrace = traceback.extract_stack()
        file, line, function, src = stackTrace[stackDepth]

        meta = self._entry.meta
        meta["facility"] = self.facility
        meta["severity"] = self.severity
        meta["filename"] = file
        meta["function"] = function
        meta["line"] = line
        meta["src"] = src
        meta["stack-trace"] = stackTrace[:stackDepth+1]

        journal.journal().record(self._entry)

        if self.fatal:
            raise self.Fatal(message)
     
        self._entry = Entry()
        return self


    def activate(self):
        self._state.set(True)
        return self


    def deactivate(self):
        self._state.set(False)
        return self

    def flip(self):
        self._state.flip()
        return self


    def __init__(self, facility, severity, state, fatal=False):
        self.facility = facility
        self.severity = severity
        
        self._entry = Entry()
        self._state = state
        self.fatal = fatal

        return


    def _getState(self):
        return self._state.get()
    

    def _setState(self, state):
        self._state.set(state)
        return
    

    state = property(_getState, _setState, None, "")


    class Fatal(Exception):


        def __init__(self, msg=""):
            self.msg = msg


        def __str__(self):
            return self.msg


# version
__id__ = "$Id: Diagnostic.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

#  End of file 
