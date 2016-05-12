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


class Parser(object):


    def locator(self):
        return self._tokenizer.locator()


    def parse(self, scanner, tokenizer):
        try:
            self._parse(scanner, tokenizer)

        except scanner.TokenizationException, error:
            msg = "-*- tokenization exception -*-"
            self._info.log(msg)
            self.onError(str(error), tokenizer.locator())

        except tokenizer.EndOfFile:
            self._info.log("-*- End of file -*-")
            self.onEndOfFile()

        return


    def __init__(self):
        self._scanner = None
        self._tokenizer = None
        return


    def _parse(self, scanner, tokenizer):

        self._scanner = scanner
        self._tokenizer = tokenizer

        done = 0
        while not done:
            token = self._tokenizer.fetch(self._scanner)
            self._info.log("token: %s" % token)

            done = token.identify(self)

        return


# version
__id__ = "$Id: Parser.py,v 1.1 2007-09-13 15:53:29 aivazis Exp $"

#  End of file 
