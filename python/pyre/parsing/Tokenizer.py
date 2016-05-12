#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2007 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


class Tokenizer(object):


    def locator(self):
        from Locator import Locator
        return Locator(self.filename, self.line, self.column)


    def fetch(self, scanner):
        """Return a token from the input stream"""

        token = None
        if self._token:
            token = self._token
            self._token = None
        else:
            token = self._tokenize(scanner)

        return token
    
        
    def unfetch(self, token):
        """Put a token back into the token stream"""

        self._token = token
        return 


    def __init__(self, file):
        self._file = file
        self._token = None

        self.filename = file.name
        self.line = 0
        self.column = 0
        self.offset = 0
        self.text = ""

        return


    # implementation

    def _tokenize(self, scanner):
        # update the column count
        self.column += self.offset

        # detect the end of line
        if self.column == len(self.text):
            self.text = self._newLine()
            self._info.log("new line: {%s}" % self.text)

        # attempt to get a token
        token = scanner.match(self.text, self.column)
        if not token:
            msg = "illegal character, could not match '%s'" % self.text[self.column:]
            raise self.TokenizationException(msg)

        # store the size of the token
        self.offset = token.size

        return token


    def _newLine(self):
        while 1:
            text = self._file.readline()
            if not text: break
            
            self.line = self.line + 1
            text = text[:-1]
            if text:
                self.column = 0
                return text
            
        raise self.EndOfFile()


    from EndOfFile import EndOfFile
    from TokenizationException import TokenizationException

    import journal
    _info = journal.debug("pyre.parsing")


# version
__id__ = "$Id: Tokenizer.py,v 1.1 2007-09-13 15:53:29 aivazis Exp $"

#  End of file 
