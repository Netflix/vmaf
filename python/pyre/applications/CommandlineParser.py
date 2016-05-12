#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


class CommandlineParser(object):


    def parse(self, root, argv=None):
        if argv is None:
            import sys
            argv = sys.argv[1:]

        return self._parse(argv, root)


    def __init__(self):
        self.help = ['?', 'h']
        self.assignment = '='
        self.prefixes = ['--', '-']
        self.separator = '.'

        import pyre.parsing.locators
        self.locator = pyre.parsing.locators.simple('command line')

        import journal
        self._debug = journal.debug("pyre.commandline")

        return


    def _parse(self, argv, root):
        help = False
        unprocessed = []

        for arg in argv:
            self._debug.line("processing '%s'" % arg)

            # is this an option
            for prefix in self.prefixes:
                if arg.startswith(prefix):
                    self._debug.line("    prefix: '%s starts with '%s'" % (arg, prefix))
                    candidate = arg[len(prefix):]
                    break
            else:
                # prefix matching failed; leave this argument alone
                self._debug.line("    prefix: '%s' is not an option" % arg)
                unprocessed.append(arg)
                continue
                
            self._debug.line("    prefix: arg='%s' after prefix stripping" % candidate)

            # skip the processing if the arg is empty after stripping
            if not candidate:
                continue

            # check for assignment
            tokens = candidate.split(self.assignment)
            self._debug.line("    tokens: %s" % `candidate`)

            # dangling =
            if len(tokens) > 1 and not tokens[1]:
                self._debug.log("tokens: bad expression: %s" % arg)
                raise CommandlineParser.CommandlineException("bad expression: '%s': no rhs" % arg)

            # lhs, rhs
            lhs = tokens[0]
            if len(tokens) > 1:
                #rhs = tokens[1]
                rhs = self.assignment.join(tokens[1:])
            else:
                rhs = "true"
            self._debug.line("    tokens: key={%s}, value={%s}" % (lhs,  rhs))

            if lhs in self.help:
                help = True
                continue

            # store this option
            self._processArgument(lhs, rhs, root)
            
        self._debug.log()

        return help, unprocessed


    def _processArgument(self, key, value, root):
        separator = self.separator
        fields = key.split(separator)
        self._debug.line("    sub: fields=%s" % fields)

        children = []
        for level, field in enumerate(fields):
            if field[0] == '[' and field[-1] == ']':
                candidates = field[1:-1].split(',')
            else:
                candidates = [field]
            self._debug.line("    sub: [%02d] candidates=%s" % (level, candidates))
            children.append(candidates)

        self._storeValue(root, children, value)

        return


    def _storeValue(self, node, children, value):
        self._debug.line("    set: children=%s" % children)
        if len(children) == 1:
            for key in children[0]:
                key = key.strip()
                self._debug.line("    option: setting '%s'='%s'" % (key, value))
                node.setProperty(key, value, self.locator)
            return

        for key in children[0]:
            self._debug.line("    sub: processing '%s'" % key)
            self._storeValue(node.getNode(key), children[1:], value)

        return


    # the class used for reporting errors
    class CommandlineException(Exception):


        def __init__(self, msg):
            self._msg = msg
            return


        def __str__(self):
            return self._msg
            

# version
__id__ = "$Id: CommandlineParser.py,v 1.2 2007-01-11 21:42:50 aivazis Exp $"

#  End of file 
