#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

import operator


class unit(object):


    _labels = ('m', 'kg', 's', 'A', 'K', 'mol', 'cd')
    _zero = (0,) * len(_labels)
    _negativeOne = (-1, ) * len(_labels)


    def __init__(self, value, derivation):
        self.value = value
        self.derivation = derivation
        return


    def __add__(self, other):
        if not self.derivation == other.derivation:
            raise IncompatibleUnits("add", self, other)

        return unit(self.value + other.value, self.derivation)


    def __sub__(self, other):
        if not self.derivation == other.derivation:
            raise IncompatibleUnits("subtract", self, other)

        return unit(self.value - other.value, self.derivation)


    def __mul__(self, other):
        if type(other) == type(0) or type(other) == type(0.0):
            return unit(other*self.value, self.derivation)
        
        value = self.value * other.value
        derivation = tuple(map(operator.add, self.derivation, other.derivation))

        if derivation == self._zero:
            return value

        return unit(value, derivation)


    def __div__(self, other):
        if type(other) == type(0) or type(other) == type(0.0):
            return unit(self.value/other, self.derivation)
        
        value = self.value / other.value
        derivation = tuple(map(operator.sub, self.derivation, other.derivation))

        if derivation == self._zero:
            return value

        return unit(value, derivation)


    def __pow__(self, other):
        if type(other) != type(0) and type(other) != type(0.0):
            raise InvalidOperation("**", self, other)

        value = self.value ** other
        derivation = tuple(map(operator.mul, [other]*7, self.derivation))

        return unit(value, derivation)
        

    def __pos__(self): return self


    def __neg__(self): return unit(-self.value, self.derivation)


    def __abs__(self): return unit(abs(self.value), self.derivation)


    def __invert__(self):
        value = 1./self.value
        derivation = tuple(map(operator.mul, self._negativeOne, self.derivation))
        return unit(value, derivation)


    def __rmul__(self, other):
        if type(other) != type(0) and type(other) != type(0.0):
            raise InvalidOperation("*", other, self)

        return unit(other*self.value, self.derivation)


    def __rdiv__(self, other):
        if type(other) != type(0) and type(other) != type(0.0):
            raise InvalidOperation("/", other, self)

        value = other/self.value
        derivation = tuple(map(operator.mul, self._negativeOne, self.derivation))
        
        return unit(value, derivation)


    def __float__(self):
        if self.derivation == self._zero: return self.value
        raise InvalidConversion(self)


    def __cmp__(self, other):
        return cmp(self.value, other.value)
            

    def __str__(self):
        str = "%g" % self.value 
        derivation = self._strDerivation()
        if not derivation:
            return str

        return str + "*" + derivation


    def __repr__(self):
        str = "%g" % self.value 
        derivation = self._strDerivation()
        if not derivation:
            return str

        return str + "*" + derivation


    def _strDerivation(self):
        return _strDerivation(self._labels, self.derivation)


# instances

one = dimensionless = unit(1, unit._zero)


# helpers
                          
def _strDerivation(labels, exponents):
    dimensions = filter(None, map(_strUnit, labels, exponents))
    return "*".join(dimensions)


def _strUnit(label, exponent):
    if exponent == 0: return None
    if exponent == 1: return label
    return label + "**%g" % exponent


# exceptions

class InvalidConversion(Exception):

    def __init__(self, operand):
        self._op = operand
        return


    def __str__(self):
        str =  "dimensional quantities ('%s') " % self._op._strDerivation()
        str = str + "cannot be converted to scalars"
        return str


class InvalidOperation(Exception):

    def __init__(self, op, operand1, operand2):
        self._op = op
        self._op1 = operand1
        self._op2 = operand2
        return


    def __str__(self):
        str = "Invalid expression: %s %s %s" % (self._op1, self._op, self._op2)
        return str


class IncompatibleUnits(Exception):

    def __init__(self, op, operand1, operand2):
        self._op = op
        self._op1 = operand1
        self._op2 = operand2
        return


    def __str__(self):
        str = "Cannot %s quanitites with units of '%s' and '%s'" % \
              (self._op, self._op1._strDerivation(), self._op2._strDerivation())
        return str
    

# version
__id__ = "$Id: unit.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#
# End of file
