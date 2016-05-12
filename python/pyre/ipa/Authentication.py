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


class Authentication(object):


    def __init__(self, username, password="", ticket=""):
        self.username = username
        self.password = password
        self.ticket = ticket
        self.action = 'login'
        return


# version
__id__ = "$Id: Authentication.py,v 1.1.1.1 2006-11-27 00:10:03 aivazis Exp $"

#  End of file 
