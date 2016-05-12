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


class Property(Trait):


    def __init__(self, name, type, default=None, meta=None, validator=None):
        Trait.__init__(self, name, type, default, meta)

        self.validator = validator

        return


    def _set(self, instance, value, locator):
        # None is a special value; it means that a property is not set
        if value is not None:
            # convert
            value = self._cast(value)
            # validate 
            if self.validator:
                value = self.validator(value)

        # record
        return Trait._set(self, instance, value, locator)


    def _getDefaultValue(self, instance):
        """retrieve the default value and return it along with a locator"""

        value = self.default

        # None is a special value and shouldn't go through the _cast
        if value is not None:
            # convert
            value = self._cast(value)
            # validate 
            if self.validator:
                value = self.validator(value)
        
        import pyre.parsing.locators
        locator = pyre.parsing.locators.simple('default')

        return value, locator


    def _cast(self, input):
        return input


    def _validate(self, value):
        return value


# version
__id__ = "$Id: Property.py,v 1.1.1.1 2006-11-27 00:10:01 aivazis Exp $"

# End of file 
