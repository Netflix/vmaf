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


from Validator import Validator


class And(Validator):


    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2
        return


    def __call__(self, candidate):
        g1 = True
        try:
            self.op1(candidate)
        except ValueError:
            g1 = False

        g2 = True
        try:
            self.op2(candidate)
        except ValueError:
            g2 = False

        if g1 and g2:
            return candidate
        
        raise ValueError("%s is not %s" % (candidate, self))


    def __str__(self):
        return "(%s and %s)" % (self.op1, self.op2)


# version
__id__ = "$Id: And.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

# End of file 
