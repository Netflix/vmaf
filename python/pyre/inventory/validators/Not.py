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


class Not(Validator):


    def __init__(self, op):
        self.op = op
        return


    def __call__(self, candidate):
        try:
            self.op(candidate)
        except ValueError:
            return candidate
        
        raise ValueError("%s is not supposed to be %s" % (candidate, self.op))


    def __str__(self):
        return "(not %s)" % (self.op)


# version
__id__ = "$Id: Not.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

# End of file 
