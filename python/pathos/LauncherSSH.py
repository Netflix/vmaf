#!/usr/bin/env python
#
## ssh Launcher class
# adapted from Mike McKerns' and June Kim's gsl SSHLauncher class
# by mmckerns@caltech.edu

"""
This module contains the derived class for secure shell (ssh) launchers
See the following for an example.


Usage
=====

A typical call to a 'ssh launcher' will roughly follow this example:

    >>> # instantiate the launcher, providing it with a unique identifier
    >>> launcher = SSH_Launcher('launcher')
    >>>
    >>> # configure the launcher to perform the command on the selected host
    >>> launcher.stage(command='hostname', rhost='remote.host.edu')
    >>>
    >>> # execute the launch and retrieve the response
    >>> launcher.launch()
    >>> print launcher.response()
 
"""
__all__ = ['LauncherSSH']

import os
import signal
import popen2
from pyre.ipc.Selector import Selector

from Launcher import Launcher
class LauncherSSH(Launcher):
    '''a remote process launcher using ssh'''

    def __init__(self, name, **kwds):
        '''create a ssh launcher

Takes one initial input:
    name        -- a unique identifier (string) for the launcher

Additional Inputs:
    rhost       -- hostname to recieve command [user@host is also valid]
    command     -- remotely launched command  [default = 'echo hello']
    launcher    -- remote service mechanism (i.e. ssh, rsh)  [default = 'ssh']
    options     -- remote service options (i.e. -v, -N, -L)  [default = '']
    fgbg        -- run in foreground/background  [default = 'foreground']

Default values are set for methods inherited from the base class:
    nodes       -- number of parallel/distributed nodes  [default = 0]
    nodelist    -- list of parallel/distributed nodes  [default = None]
        '''
       #Launcher.__init__(self, name)
        super(LauncherSSH, self).__init__(name)
        self.stage(**kwds)
        return

    class Inventory(Launcher.Inventory):
        import pyre.inventory

        launcher = pyre.inventory.str('launcher', default='ssh')
        options = pyre.inventory.str('options', default='')
        rhost = pyre.inventory.str('rhost', default='localhost')
        command = pyre.inventory.str('command', default='echo hello')
        fgbg = pyre.inventory.str('fgbg', default='foreground')
       #XXX: also inherits 'nodes' and 'nodelist'
        pass

   #def _configure(self):
   #    #FIXME: bypassing this with 'stage'
   #    return

    def stage(self, **kwds):
        '''stage a remote command using given keywords:

(Re)configure the copier for the following inputs:
    rhost       -- hostname to recieve command [user@host is also valid]
    command     -- remotely launched command  [default = 'echo hello']
    launcher    -- remote service mechanism (i.e. ssh, rsh)  [default = 'ssh']
    options     -- remote service options (i.e. -v, -N, -L)  [default = '']
    fgbg        -- run in foreground/background  [default = 'foreground']
        '''
        for key, value in kwds.items():
            if key == 'command':
                self.inventory.command = value
            elif key == 'rhost':
                self.inventory.rhost = value
            elif key == 'launcher':
                self.inventory.launcher = value
            elif key == 'options':
                self.inventory.options = value
            elif key == 'fgbg':
                self.inventory.fgbg = value
        return

    def launch(self):
        '''launch a staged command'''
        command = '%s %s %s "%s"' % (self.inventory.launcher,
                                   self.inventory.options,
                                   self.inventory.rhost,
                                   self.inventory.command)
       #self._execStrategy(command)
        self._execute(command)
        return

    def _execute(self, command):
       #'''execute the launch by piping the command, & saving the file object'''
        if self.inventory.fgbg in ['foreground','fg']:
            f = os.popen(command, 'r')
            self._fromchild = f #save fileobject
            self._pid = 0 #XXX: MMM --> or -1 ?
        else: #Spawn an ssh process 
            p = popen2.Popen4(command)
            self._pid = p.pid #get fileobject pid
            self._fromchild = p.fromchild #save fileobject
        return

    def response(self):
        '''read the response from remotely launched process'''

        self._response = ''
        
        def onData(selector, fobj):
            self._debug.log('on_remote')
            self._response = fobj.readline()
            selector.state = False
            return

        def onTimeout(selector):
            selector.state = False
        
        sel = Selector()
        #sel._info.activate()
        sel.notifyOnReadReady(self._fromchild, onData)
        sel.notifyWhenIdle(onTimeout)
        sel.watch(2.0)
        return self._response

    def pid(self):
        '''get launcher pid'''
        return self._pid

    def kill(self):
        '''terminate the launcher'''
        if self._pid > 0:
            print 'Kill ssh pid=%d' % self._pid
            os.kill(self._pid, signal.SIGTERM)
            os.waitpid(self._pid, 0)
            self._pid = 0
        return
    pass


if __name__ == '__main__':
    pass


# End of file
