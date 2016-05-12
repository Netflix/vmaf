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


import pythlets
_defaultResourceFile = pythlets.resourceFile("ctree.glade")


from SimpleRenderer import SimpleRenderer


class Explorer(SimpleRenderer):


    # Explorer only visits directory nodes
    def onDirectory(self, node):

        parent = self._currentNode
        children = node.subdirectories()

        isExpanded = 0
        isLeaf = not children

        self._info.log(" visitor: directory={%s}, isLeaf=%d" % (node.path, isLeaf))
        
        newNode = self._tree.insert_node(
            parent, None, [node.name], 5,
            self._folderImage, self._folderMask,
            self._openFolderImage, self._openFolderMask,
            isLeaf, isExpanded
            )

        self._nodemap[newNode] = (node, 0)

        self._currentNode = newNode
        for child in node.subdirectories():
            child.id(self)

        self._currentNode = parent
        
        return


    def __init__(self, resourceFile=_defaultResourceFile):
        import gtk

        self._currentNode = None
        self._panel = self._constructPanel(resourceFile)

        self._tree = self._panel.get_widget("ctree")
        self._folderImage, self._folderMask = \
                           gtk.create_pixmap_from_xpm(self._tree, None, folder)
        self._openFolderImage, self._openFolderMask = \
                           gtk.create_pixmap_from_xpm(self._tree, None, open_folder)


        self._nodemap = {}

        return
    

    def _constructPanel(self, filename):
        import gtk
        import GTK
        import libglade

        panel = libglade.GladeXML(filename)

        # connect the signals
        panel.signal_connect('on_exit', gtk.mainquit)
        panel.signal_connect('on_expand', self._expand)
        panel.signal_connect('on_collapse', self._collapse)
        panel.signal_connect('on_select_row', self._select)

        ctree = panel.get_widget("ctree")
        ctree.set_indent(17)
        ctree.set_row_height(18)
        ctree.set_line_style(GTK.CTREE_LINES_DOTTED)
        ctree.set_expander_style(GTK.CTREE_EXPANDER_SQUARE)

        return panel


    def _expand(self, tree, node):
        directory = self._nodemap[node][0].path
        self._info.log("  expand: directory={%s}" % directory)
        return


    def _collapse(self, tree, node):
        directory = self._nodemap[node][0].path
        self._info.log("collapse: directory={%s}" % directory)
        return


    def _select(self, tree, node, column):
        if (column >= 0):
            directory = self._nodemap[node][0].path
            self._info.log("  select: directory={%s}" % directory)

        return


# images

folder = pythlets.pixmap("folder-closed.xpm")
open_folder = pythlets.pixmap("folder-open.xpm")


# version
__id__ = "$Id: Explorer.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

#  End of file 
