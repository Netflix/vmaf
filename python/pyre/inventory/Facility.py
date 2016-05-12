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


from Trait import Trait


class Facility(Trait):


    def __init__(self, name, family=None, default=None, factory=None, args=(), meta=None):
        Trait.__init__(self, name, 'facility', default, meta)

        self.args = args
        self.factory = factory

        if family is None:
            family = name
        self.family = family

        return


    def _getDefaultValue(self, instance):
        component = self.default

        # build a default locator
        import pyre.parsing.locators
        here = pyre.parsing.locators.simple('default')

        if component is not None:
            # if we got a string, resolve
            if isinstance(component, basestring):
                component, locator = self._retrieveComponent(instance, component, args=())
                here = pyre.parsing.locators.chain(locator, here)
                
            return component, here

        if self.factory is not None:
            # instantiate the component
            component =  self.factory(*self.args)
            # adjust the configuration aliases to include my name
            aliases = component.aliases
            if self.name not in aliases:
                aliases.append(self.name)
            
            # build a default locator
            import pyre.parsing.locators
            locator = pyre.parsing.locators.simple('default')
            # return
            return component, locator

        # oops: expect exceptions galore!
        import journal
        firewall = journal.firewall('pyre.inventory')
        firewall.log(
            "facility %r was given neither a default value nor a factory method" % self.name)
        return None, None


    def _set(self, instance, component, locator):
        if isinstance(component, basestring):
            try:
                name, args = component.split(":")
                args = args.split(",")
            except ValueError:
                name = component
                args = []
                
            component, source = self._retrieveComponent(instance, name, args)

            import pyre.parsing.locators
            locator = pyre.parsing.locators.chain(source, locator)

        if component is None:
            try:
                descriptor = instance._getTraitDescriptor(self.name)
            except KeyError:
                descriptor = instance._createTraitDescriptor()
                instance._setTraitDescriptor(self.name, descriptor)
            descriptor.value = None
            descriptor.locator = locator
            descriptor.inquiry = '%s:%s' % (name, args)
            return

        # get the old component
        try:
            old = instance._getTraitValue(self.name)
        except KeyError:
            # the binding was uninitialized
            return instance._initializeTraitValue(self.name, component, locator)

        # if the previous binding was non-null, finalize it
        if old:
            old.fini()
        
        # bind the new value
        return instance._setTraitValue(self.name, component, locator)


    def _retrieveComponent(self, instance, componentName, args):
        component = instance.retrieveComponent(name=componentName, factory=self.family, args=args)

        if component is not None:
            locator = component.getLocator()
        else:
            import pyre.parsing.locators
            component = self._import(componentName)

            if component:
                locator = pyre.parsing.locators.simple('imported')
            else:
                locator = pyre.parsing.locators.simple('not found')
                return None, locator

        # adjust the names by which this component is known
        component.aliases.append(self.name)
            
        return component, locator


    def _import(self, name):
        try:
            module = __import__(name, {}, {})
        except ImportError:
            import traceback
            tb = traceback.format_exc()
            
            import journal
            journal.error("pyre.inventory").log(
                "could not bind facility '%s': component '%s' not found:\n%s" % (
                self.name, name, tb)
                )
            return

        try:
            factory = module.__dict__[self.family]
        except KeyError:
            import journal
            journal.error("pyre.inventory").log(
                "no factory for facility '%s' in '%s'" % (self.name, module.__file__))
            return

        try:
            item = factory(*self.args)
        except TypeError:
            import journal
            journal.error("pyre.inventory").log(
                "no factory for facility '%s' in '%s'" % (self.name, module.__file__))
            return

        return item


    # interface registry
    _interfaceRegistry = {}

    # metaclass
    from Interface import Interface
    __metaclass__ = Interface


# version
__id__ = "$Id: Facility.py,v 1.1.1.1 2006-11-27 00:10:00 aivazis Exp $"

# End of file 
