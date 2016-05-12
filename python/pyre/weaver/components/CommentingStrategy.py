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


class CommentingStrategy(object):


    def commentBlock(self, lines):
        block = []

        if not lines:
            return block

        block.append(self._beginCommentBlock(lines[0]))

        for line in lines[1:]:
            block.append(self._commentLineInBlock(line))
        block.append(self._endCommentBlock())

        return block


    def line(self, line=''):
        raise NotImplementedError(
            "class '%s' should override 'line'" % self.__class__.__name__)


    def __init__(self):
        return


    def _beginCommentBlock(self, text=''):
        raise NotImplementedError(
            "class '%s' should override '_beginCommentBlock'"
            % self.__class__.__name__)
                                  

    def _commentLineInBlock(self, line=''):
        raise NotImplementedError(
            "class '%s' should override '_commentLineInBlock'"
            % self.__class__.__name__)


    def _endCommentBlock(self, text=''):
        raise NotImplementedError(
            "class '%s' should override '_beginCommentBlock'"
            % self.__class__.__name__)


# version
__id__ = "$Id: CommentingStrategy.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
