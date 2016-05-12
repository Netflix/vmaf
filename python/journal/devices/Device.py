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


class Device(object):


    def record(self, entry):
        text = self.renderer.render(entry)
        self._write(text)
        return


    def __init__(self, renderer=None):
        if renderer is None:
            from Renderer import Renderer
            renderer = Renderer()

        self.renderer = renderer

        return


    def _write(self, entry):
        raise NotImplementedError("class '%s' must override '_write'" % self.__class__.__name__)


# version
__id__ = "$Id: Device.py,v 1.1.1.1 2006-11-27 00:09:35 aivazis Exp $"

#  End of file 
