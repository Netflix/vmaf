#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


from Device import Device


class File(Device):


    def __init__(self, logfile):
        Device.__init__(self)
        self.file = logfile
        return


    def _write(self, message):
        for line in message:
            print >> self.file, line

        self.file.flush()

        return


# version
__id__ = "$Id: File.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

#  End of file 
