#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


class FileSystem(object):

    def root(self):
        return self._root


    def expand(self, levels=0):
        level = 0
        todo = [self._root]

        while todo:
            working = todo
            todo = []
            for directory in working:
                todo += directory.expand()

            level += 1
            if levels and level >= levels: break

        return
        

    def __init__(self, root):
        import os
        from Root import Root

        directory = os.path.abspath(root)
        self._root = Root(directory) 

        return


# version
__id__ = "$Id: FileSystem.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

#  End of file 
