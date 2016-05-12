#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.components.Component import Component


class Weaver(Component):


    # inventory
    class Inventory(Component.Inventory):

        import pyre.inventory

        author = pyre.inventory.str("author", default="")
        organization = pyre.inventory.str("organization", default="")
        copyright = pyre.inventory.str("copyright", default="")

        bannerWidth = pyre.inventory.int("bannerWidth", default=78)
        bannerCharacter = pyre.inventory.str("bannerCharacter", default='~')

        creator = pyre.inventory.str("creator")
        timestamp = pyre.inventory.bool("timestamp", default=True)

        lastLine = pyre.inventory.str("lastLine", default=" End of file ")
        copyrightLine = pyre.inventory.str(
            "copyrightLine", default="(C) %s  All Rights Reserved")
        licenseText = pyre.inventory.preformatted("licenseText", default=["{LicenseText}"])
        
        timestampLine = pyre.inventory.str(
            "timestampLine", default=" Generated automatically by %s on %s")

        versionId = pyre.inventory.str("versionId", default=' $' + 'Id' + '$')
    

    def weave(self, document=None, stream=None):
        # produce the text
        text = self.render(document)

        # verify the output stream
        if stream is None:
            import sys
            stream = sys.stdout
        print >> stream, "\n".join(text)

        return


    def splice(self, body):
        self.begin()
        self.contents(body)
        self.end()
        return self.document()
        

    def render(self, document=None):
        self._renderer.options = self.inventory
        return self._renderer.weave(document)


    def begin(self):
        self._renderer.options = self.inventory
        self._renderer.begin()
        return


    def contents(self, body):
        self._renderer.contents(body)
        return


    def end(self):
        self._renderer.end()
        return


    def document(self):
        return self._renderer.document()


    def languages(self):
        candidates = self.inventory.retrieveShelves(address=['mills'], extension='odb')
        
        candidates.sort()
        return candidates


    def __init__(self, name=None):
        if name is None:
            name = 'weaver'
            
        Component.__init__(self, name, facility='weaver')

        self._renderer = None
        self._language = None
        
        return


    # language property
    def _getLanguage(self):
        return self._language


    def _setLanguage(self, language):
        self._language = language
        self._renderer = self._retrieveLanguage(language)
        return


    def _retrieveLanguage(self, language):
        weaver = self.retrieveComponent(
            factory=self.name, name=language, vault=['mills'])

        if weaver:
            return weaver
                    
        import journal
        journal.error('pyre.weaver').log("could not locate weaver for '%s'" % language)

        self.getCurator().dump()

        return None


    language = property(_getLanguage, _setLanguage, None, "")
        

    # renderer property
    def _getRenderer(self):
        return self._renderer


    def _setRenderer(self, renderer):
        self._renderer = renderer
        return

    renderer = property(_getRenderer, _setRenderer, None, "")
        

# version
__id__ = "$Id: Weaver.py,v 1.2 2007-03-06 04:16:38 aivazis Exp $"

# End of file 
