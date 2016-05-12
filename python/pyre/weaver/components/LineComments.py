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

from CommentingStrategy import CommentingStrategy


class LineComments(CommentingStrategy):


    def line(self, line=""):
        return self.commentLine + line


    def _beginCommentBlock(self, text=''):
        return self.commentLine + text


    def _commentLineInBlock(self, line=''):
        return self.line(line)


    def _endCommentBlock(self, text=''):
        return self.commentLine + text


# version
__id__ = "$Id: LineComments.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
