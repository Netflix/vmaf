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


class Renderer(object):


    def render(self, entry):

        text = []
        meta = entry.meta

        if self.header:
            filename = meta["filename"]
            if self.trimFilename and len(filename) > 53:
                filename = filename[0:20] + "..." + filename[-30:]
                meta["filename"] = filename
            text.append(self.header % meta)

        for line in entry.text:
            text.append(self.format % line)

        if self.footer:
            text.append(self.footer % meta)

        return text


    def __init__(self, header=None, format=None, footer=None):
        if header is None:
            header = " >> %(filename)s:%(line)s:%(function)s\n >> %(facility)s(%(severity)s)"

        if format is None:
            format = " -- %s"

        if footer is None:
            footer = ""

        self.header = header
        self.format = format
        self.footer = footer

        self.trimFilename = False

        return


# version
__id__ = "$Id: Renderer.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

#  End of file 
