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


from Device import Device


class File(Device):


    class Inventory(Device.Inventory):

        import pyre.inventory
        
        name = pyre.inventory.str("name", default="journal.log")
        name.meta['tip'] = "the name of the file in which messages will be placed"


    def createDevice(self):
        logfile = file(self.inventory.name, "a", 0)

        import os
        import time
        
        print >> logfile, " ** MARK: opened by %s on %s" % (os.getpid(), time.ctime())

        from journal.devices.File import File
        return File(logfile)


    def __init__(self):
        Device.__init__(self, "file")
        return


# version
__id__ = "$Id: File.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

# End of file 
