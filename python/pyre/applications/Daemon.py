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


from Stager import Stager


class Daemon(Stager):


    def execute(self, *args, **kwds):
        self.args = args
        self.kwds = kwds

        try:
            spawn = self.kwds['spawn']
        except KeyError:
            spawn = True
        
        if not spawn:
            print " ** daemon %r in debug mode" % self.name
            self.daemon(0)
            return
            
        import pyre.util
        return pyre.util.spawn(self.done, self.respawn)


    def done(self, pid):
        return


    def respawn(self, pid):
        import os
        os.chdir("/")
        os.setsid()
        os.umask(0)

        import pyre.util
        pyre.util.spawn(self.exit, self.daemon)
        
        return


    def exit(self, pid):
        import sys
        sys.exit(0)

        # unreachable
        import journal
        journal.firewall("pyre.services").log("UNREACHABLE")
        return
        

    def daemon(self, pid):
        import os

        # change the working directory to my home directory
        if not os.path.exists(self.home):
            import journal
            journal.error(self.name).log("directory %r does not exist" % self.home)
            self.home = '/tmp'

        os.chdir(self.home)

        # redirect the journal output since we are about to close all the
        # standard file descriptors
        # currently disabled since a better strategy is to have the application author
        # build a journal configuration file
        # self.configureJournal()

        # close all ties with the parent process, unless in debug mode
        if pid:
            os.close(2)
            os.close(1)
            os.close(0)

        # launch the application
        self.main(*self.args, **self.kwds)

        return


    def configureJournal(self):
        # open the logfile
        stream = file(self.name + '.log', "w")

        # attach it as the journal device
        import journal
        journal.logfile(stream)

        return
        


    def __init__(self):
        self.args = ()
        self.kwds = {}

        self.home = '/tmp'

        return


# version
__id__ = "$Id: Daemon.py,v 1.2 2008-01-31 15:47:35 aivazis Exp $"

# End of file 
