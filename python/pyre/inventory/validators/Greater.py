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


from Binary import Binary


class Greater(Binary):


    def __call__(self, candidate):
        if candidate > self.value:
            return candidate

        raise ValueError("%s is not %s" % (candidate, self))


    def __str__(self):
        return "(greater than %s)" % (self.value)


# version
__id__ = "$Id: Greater.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

# End of file 
