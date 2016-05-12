#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
"""
This module contains the base class for pathos launchers, and describes
the pathos launcher interface. The 'launch' method must be overwritten
with the derived launcher's execution algorithm. See the following for
an example.


Usage
=====

A typical call to a pathos 'launcher' will roughly follow this example:

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
__all__ = ['Launcher']

from pyre.components.Component import Component


class Launcher(Component):
    """
Base class for launchers for parallel and distributed computing.
    """

    class Inventory(Component.Inventory):

        import pyre.inventory

        nodes = pyre.inventory.int("nodes", default=0)
        nodelist = pyre.inventory.slice("nodelist")


    def launch(self):
        """execute the launcher

*** this method must be overwritten ***"""
        raise NotImplementedError("class '%s' must override 'launch'" % self.__class__.__name__)


    def __init__(self, name, facility="launcher"):
        """
Takes one initial input:
    name     -- a unique identifier (string) for the launcher

Additionally, default values are set for 'inventory' class members:
    nodes    -- number of parallel/distributed nodes  [default = 0]
    nodelist -- list of parallel/distributed nodes  [default = None]
        """
       #Component.__init__(self, name, facility="launcher")
        super(Launcher, self).__init__(name, facility)
        self.nodes = 0
        self.nodelist = None
        return


    def _configure(self):
        self.nodes = self.inventory.nodes
        self.nodelist = self.inventory.nodelist
        return


# version
# file copied from pythia-0.8 pyre.mpi.Launcher.py (svn:danse.us/pyre -r2)

# End of file 
