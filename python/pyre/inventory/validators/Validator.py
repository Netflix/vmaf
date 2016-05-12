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


class Validator(object):


    def __call__(self, candidate):
        raise NotImplementedError("class '%s' must override '__call__'" % self.__class__.__name__)


# version
__id__ = "$Id: Validator.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

# End of file 
