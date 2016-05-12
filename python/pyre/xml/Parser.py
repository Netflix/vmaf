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


import xml.sax


class Parser(xml.sax.ContentHandler):


    # parsing
    def parse(self, stream, document, parserFactory=None):

        self._init(document)
        self._parse(stream, parserFactory)
        self._fini()

        return self._document.document


    # content demultiplexing

    def startDocument(self):
        line, column = self._locator.getLineNumber(), self._locator.getColumnNumber()
        self._info.log("startDocument at (%d, %d)" % (line, column))

        # give the document node a way to get at line info
        self._document.locator = self._locator
        return


    def endDocument(self):
        line, column = self._locator.getLineNumber(), self._locator.getColumnNumber()
        self._info.log("endDocument at (%d, %d)" % (line, column))

        if self._document != self._currentNode:
            import journal
            journal.firewall("pyre.xml.parsing").log("ooooops!")

        # break a circular reference introduced above
        self._document.locator = None

        return

        
    def startElement(self, name, attributes):
        line, column = self._locator.getLineNumber(), self._locator.getColumnNumber()
        self._info.log("startElement: '%s', at (%d, %d)" % (name, line, column))
        self._nodeStack.append(self._currentNode)
        self._currentNode = self._document.node(name, attributes)

        return


    def characters(self, content):
        if content: 
            line, column = self._locator.getLineNumber(), self._locator.getColumnNumber()
            self._info.log("characters: '%s', at (%d, %d)" % (content, line, column))
            self._currentNode.content(content)

        return


    def endElement(self, name):
        line, column = self._locator.getLineNumber(), self._locator.getColumnNumber()
        self._info.log("endElement: '%s', at (%d, %d)" % (name, line, column))

        node = self._currentNode
        self._currentNode = self._nodeStack.pop()

        try:
            node.notify(self._currentNode)
        except ValueError, text:
            l = self._document.locator

            import journal
            error = journal.error("pyre.xml.parsing")
            error.log("%s: line %s, column %s: %s" % (l.filename, l.line, l.column, text))

        return


    def processingInstruction(self, target, data):
        import journal
        journal.firewall("pyre.xml.parsing").log(
            "processingInstruction: target={%s}, data={%s}" % (target, data)
            )
        
        return


    # constructor
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

        self._document= None
        self._nodeStack = []
        self._currentNode = None

        return


    def _init(self, document):
        self._nodeStack = []
        self._currentNode = document
        self._document = document
        return


    def _parse(self, stream, factory):
        # create a parser
        if factory:
            parser = factory()
        else:
            parser = xml.sax.make_parser()

        # parse
        parser.setContentHandler(self)
        parser.parse(stream)
        parser.setContentHandler(None)

        return


    def _fini(self):
        self._nodeStack = []
        self._currentNode = None
        return


    import journal
    _info = journal.debug("pyre.xml.parsing")
    del journal


# version
__id__ = "$Id: Parser.py,v 1.1.1.1 2006-11-27 00:10:10 aivazis Exp $"

#  End of file 
