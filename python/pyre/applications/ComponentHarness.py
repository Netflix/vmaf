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


class ComponentHarness(object):


    def harnessComponent(self):
        """harness an external component"""

        # create the component
        component = self.createComponent()

        # initialize the persistent store used by the component to configure itself
        curator = self.prepareComponentCurator()

        # prepare optional configuration for the component
        registry = self.prepareComponentConfiguration(component)

        # configure the component
        # collect unknown traits for the components and its subcomponents
        up, uc = self.configureHarnessedComponent(component, curator, registry)

        if not self.verifyConfiguration(up, uc):
            return

        # initialize the component
        component.init()

        # register it
        self.component = component

        return component


    def fini(self):
        """finalize the component"""
        
        if self.component:
            self.component.fini()

        return


    def createComponent(self):
        """create the harnessed component"""
        raise NotImplementedError(
            "class %r must override 'createComponent'" % self.__class__.__name__)


    def prepareComponentCurator(self):
        """prepare the persistent store manager for the harnessed component"""

        # by default, assume that this is a mixin class and the host has a
        # notion of its persistent store that it wants to share with the
        # harnessed component
        return self.getCurator()
        

    def prepareComponentConfiguration(self, component):
        """prepare the persistent store manager for the harnessed component"""

        # by default, assume that this is a mixin class and the host has a
        # registry with settings for the harnessed component
        registry = self.pruneRegistry()
        registry.name = component.name

        return registry


    def configureHarnessedComponent(self, component, curator, registry):
        """configure the harnessed component"""

        # link the component with the curator
        component.setCurator(curator)
        component.initializeConfiguration()

        # update the component's inventory with the optional settings we
        # have gathered on its behalf
        component.updateConfiguration(registry)

        # load the configuration onto the inventory
        up, uc = component.applyConfiguration()

        # return the rejected settings
        return up, uc


    def __init__(self):
        self.component = None
        return


# version
__id__ = "$Id: ComponentHarness.py,v 1.1.1.1 2006-11-27 00:09:54 aivazis Exp $"

# End of file 
