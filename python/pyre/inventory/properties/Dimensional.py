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


from pyre.inventory.Property import Property


class Dimensional(Property):


    def __init__(self, name, default=0.0, meta=None, validator=None):
        Property.__init__(self, name, "dimensional", default, meta, validator)

        try:
            self.len = len(default)
        except TypeError:
            self.len = 0
            
        return


    def _cast(self, value):
        candidate = value
        if isinstance(value, basestring):
            import pyre.units
            parser = pyre.units.parser()
            candidate = parser.parse(value)

        self._checkDimensions(candidate, value)

        return candidate


    def _checkDimensions(self, candidate, setting):
        try:
            size = len(candidate)
        except TypeError:
            size = 0
        
        if size != self.len:
            raise ValueError, "value '%s' is not the same shape as the default '%s'" % (
                setting, self.default)
        
        if self.len == 0:
            tokens = [candidate]
            target = [self.default]
        else:
            tokens = candidate
            target = self.default

        from pyre.units.unit import unit
        for a,b in zip(tokens, target):
            if not isinstance(a, unit) and not isinstance(b, unit):
                continue
            
            if isinstance(a, unit) and not isinstance(b, unit):
                raise ValueError, "dimension mismatch between input '%s' and target '%s'" % (
                    setting, self.default)

            if not isinstance(a, unit) and isinstance(b, unit):
                raise ValueError, "dimension mismatch between input '%s' and target '%s'" % (
                    setting, self.default)

            if a.derivation != b.derivation:
                raise ValueError, "dimension mismatch between input '%s' and target '%s'" % (
                    setting, self.default)

        return


# version
__id__ = "$Id: Dimensional.py,v 1.1.1.1 2006-11-27 00:10:02 aivazis Exp $"

# End of file 
