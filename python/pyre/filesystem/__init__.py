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

# factories

def filesystem(name):
    from FileSystem import FileSystem
    return FileSystem(name)


def root(name):
    from Root import Root
    return Root(name)


def directory(name, parent):
    from Directory import Directory
    return Directory(name, parent)


def file(name, parent):
    from File import File
    return File(name, parent)


# methods

def listing(fs):
    from SimpleRenderer import SimpleRenderer
    renderer = SimpleRenderer()
    renderer.render(fs.root())
    return


def explore(fs):
    from Explorer import Explorer
    
    renderer = Explorer()
    renderer.render(fs.root())

    gtk.mainloop()

    return


def tree(fs):
    from TreeRenderer import TreeRenderer

    renderer = TreeRenderer()
    renderer.render(fs.root())

    return


def find(fs, name):
    from Finder import Finder

    root = fs.root()
    finder = Finder()

    return finder.find(root, name)


from fastfind import fastfind


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

#  End of file 
