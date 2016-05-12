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


from File import File


class Directory(File):


    def identify(self, inspector):
        return inspector.onDirectory(self)


    def children(self):
        return tuple(self._children.values())


    def files(self):
        return tuple(self._files)


    def subdirectories(self):
        return tuple(self._subdirectories)


    def expand(self):

        import os
        import stat
        from BlockDevice import BlockDevice
        from CharacterDevice import CharacterDevice
        from File import File
        from Link import Link
        from NamedPipe import NamedPipe
        from Socket import Socket

        import journal
        debug = journal.debug("pyre.filesystem")

        files = []
        subdirectories = []

        root = self.path
        children = os.listdir(root)
        debug.log("directory '%s' has %d files" % (self.name, len(children)))

        count = 0
        for name in children:
            count += 1

            if name in self._children:
                continue
            
            pathname = os.path.join(root, name)
            # PORTABILITY: lstat is unix only
            mode = os.lstat(pathname)[stat.ST_MODE]

            if stat.S_ISDIR(mode):
                node = Directory(name, self)
                subdirectories.append(node)
            elif stat.S_ISREG(mode):
                node = File(name, self)
                files.append(node)
            elif stat.S_ISLNK(mode):
                node = Link(name, self)
            elif stat.S_ISSOCK(mode):
                node = Socket(name, self)
            elif stat.S_ISFIFO(mode):
                node = NamedPipe(name, self)
            elif stat.S_ISCHR(mode):
                node = CharacterDevice(name, self)
            elif stat.S_ISBLK(mode):
                node = BlockDevice(name, self)
            else:
                Firewall.hit("unknown file type: mode=%x" % mode)

            self._children[node.name] = node

            if not count % 1000:
                debug.log("processed %d files" % count)

        debug.log("total files processed: %d" % count)

        self._files = files
        self._subdirectories = subdirectories

        return subdirectories


    def __init__(self, name, parent):
        File.__init__(self, name, parent)

        self._children = {}

        self._files = []
        self._subdirectories = []

        return


# version
__id__ = "$Id: Directory.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

#  End of file 
