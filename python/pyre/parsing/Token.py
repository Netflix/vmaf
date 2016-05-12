#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2007 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


class Token(object):


    def __init__(self, match, groups):
        self.match = match
        self.lexeme = groups[self.__class__.__name__]
        self.size = len(self.lexeme)
        return


    def __str__(self):
        return "{token: %s}" % self.lexeme


    __slots__ = ("lexeme", "match", "size")


# version
__id__ = "$Id: Token.py,v 1.1 2007-09-13 15:53:29 aivazis Exp $"

#  End of file 
