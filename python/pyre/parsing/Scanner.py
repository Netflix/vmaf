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


class Scanner(object):


    def match(self, text, column):
        match = self._scanner.match(text, column)
        if not match:
            return None

        return self._decode(match)
            

    def __init__(self):
        self._tokens = self._tokenClasses()
        self._scanner = self._constructTokenRecognizer()
        return


    def _tokenClasses(self):
        raise NotImplementedError(
            "class '%s' must override '_tokenClasses'" % self.__class__.__name__)


    def _constructTokenRecognizer(self):
        import re
        patterns = [
            "(?P<%s>%s)" % (token.__name__, token.pattern)
            for token in self._tokens]

        scanner = '|'.join(patterns)
        self._debugPattern.log("pattern: {%s}" % scanner)
        return re.compile(scanner)


    def _decode(self, match):
        groups = match.groupdict()
        for token in self._tokens:
            if groups[token.__name__]:
                self._info.log("matched token class '%s'" % token.__name__)
                return token(match, groups)

        str = "The text '%s' matched the scanner pattern " % match.group()
        str += "but there is no corresponding token class"
        import journal
        journal.firewall("pyre.parsing").log(str)

        return


    def _pattern(self):
        return self._scanner.pattern


    import journal
    _info = journal.debug("pyre.parsing.scanner")
    _debugPattern = journal.debug("pyre.parsing.scanner.pattern")


    from TokenizationException import TokenizationException

            
# version
__id__ = "$Id: Scanner.py,v 1.1 2007-09-13 15:53:29 aivazis Exp $"

#  End of file 
