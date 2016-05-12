#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class AbstractNode(object):


    # abstract methods
    # parser.characters handler
    def content(self, text):
        return


    # parser.endElement handler
    def notify(self, target):
        raise NotImplementedError(
            "class '%s' should override method 'notify'" % self.__class__.__name__)


    # the default constructor is also abstract, hence useless
    # descendants must override and process the tag attributes
    def __init__(self, document):
        self.document = document
        return


# version
__id__ = "$Id: AbstractNode.py,v 1.1.1.1 2006-11-27 00:10:09 aivazis Exp $"

# End of file 
