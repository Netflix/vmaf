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


class BlockComments(CommentingStrategy):


    def line(self, line=""):
        return self.commentBeginBlock + line + self.commentEndBlock


    def _beginCommentBlock(self, text=''):
        return self.commentBeginBlock + text


    def _commentLineInBlock(self, line=''):
        return self.commentBlockLine + line


    def _endCommentBlock(self, text=''):
        return self.commentEndBlock + text


# version
__id__ = "$Id: BlockComments.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
