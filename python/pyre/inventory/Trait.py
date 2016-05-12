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


class Trait(object):


    def __init__(self, name, type, default=None, meta=None):
        self.name = name
        self.default = default

        # the printable type tag
        self.type = type

        # the default meta is a dictionary of user supplied information
        if meta is None:
            meta = {}
            
        self.meta = meta
        return


    def __get__(self, instance, cls=None):

        # attempt to get hold of the instance's attribute record
        try:
            return instance._getTraitValue(self.name)

        # instance is None when accessed as a class variable
        except AttributeError:
            # catch bad descriptors or changes in the python conventions
            if instance is not None:
                import journal
                firewall = journal.firewall("pyre.inventory")
                firewall.log("AttributeError on non-None instance. Bad descriptor?")

            # interpret this usage as a request for the trait object itself
            return self

        except KeyError:
            # the value of this trait in this instance is uninitialized
            # initialize it and return the default value
            return self._initialize(instance)

        # not reachable
        return None


    def __set__(self, instance, value):
        import traceback
        stack = traceback.extract_stack()
        source, line, function, text = stack[-2]

        import pyre.parsing.locators
        locator = pyre.parsing.locators.script(source, line, function)

        self._set(instance, value, locator)
        
        return


    def _getDefaultValue(self, instance):
        """retrieve the default value and return it along with a locator"""
        raise NotImplementedError(
            "class %r must override '_getDefaultValue" % self.__class__.__name__)


    def _initialize(self, instance):
        # obtain the default value -- descendants must define this
        value, locator = self._getDefaultValue(instance)
        instance._initializeTraitValue(self.name, value, locator)

        # return the default value
        return value


    def _set(self, instance, value, locator):
        try:
            return instance._setTraitValue(self.name, value, locator)
        except KeyError:
            return instance._initializeTraitValue(self.name, value, locator)

        # UNREACHABLE
        import journal
        journal.firewall("pyre.inventory").log("UNREACHABLE")
        return


# version
__id__ = "$Id: Trait.py,v 1.1.1.1 2006-11-27 00:10:01 aivazis Exp $"

# End of file 
