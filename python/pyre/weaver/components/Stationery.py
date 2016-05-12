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



class Stationery(object):


    def header(self):

        options = self._options
        separator = self.separator()

        licenseText = self._formatLicense()

        h = ['', separator, '']
        h += self.copyright()
        h += licenseText
        h += ['', separator]

        return [self.firstLine] + self.commentBlock(h)


    def footer(self):
        f = ['', self.line(self._options.lastLine) ]
        return f
            

    def copyright(self):
        c = []
        options = self._options

        # required
        width = options.bannerWidth

        # optional
        author = options.author
        copyright = options.copyright
        organization = options.organization

        if author:
            c.append(author.center(width).rstrip())

        if organization:
            c.append(organization.center(width).rstrip())

        if copyright:
            c.append((options.copyrightLine % copyright).center(width).rstrip())

        return c


    def separator(self):
        options = self._options
        banner = options.bannerCharacter
        cycles = options.bannerWidth/len(banner)
        separator = ' ' + banner * cycles
        return separator


    def blankLine(self):
        return ''


    def __init__(self, name):
        self._name = name
        self._options = None

        import journal
        self._debug = journal.debug(name)

        return


    def _formatLicense(self):
        substitutions = {
            'ORGANIZATION': self._options.organization,
            }
        raw = self._options.licenseText
        if len(raw) < 2 and not raw[0]:
            return []
        
        import pyre.util
        result = ['']
        result += [ " %s" % pyre.util.expandMacros(line, substitutions) for line in raw ]
        
        return result


    def _getOptions(self):
        return self._options


    def _setOptions(self, options):
        self._options = options
        return


    options = property(_getOptions, _setOptions, None, "")


# version
__id__ = "$Id: Stationery.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
