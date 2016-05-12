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


class DTDBuilder(type):

    CALLBACK_CHECK = 'relaxed'


    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        trash = {}

        # initialize the lookup table with inherited tags
        dtd = {}
        bases = list(bases)
        bases.reverse()
        for base in bases:
            try:
                dtd.update(base._mydtd)
            except AttributeError:
                pass

        # process the list of nested tag factories
        for node in dict.get('tags', []):
            symbols = {}

            # parse out any explicit package specs
            nsplit = node.split('.')
            factory = nsplit[-1]

            # build the path to the tag module
            # if there is no explicit package, build a path to the location of this node
            if len(nsplit) == 1:
                path = cls.__module__.split('.')[:-1]
                path.append(node)
                module = '.'.join(path)
            else:
                # respect the explicit path supplied
                module = node
            
            # attempt to load the node constructor
            symbols = __import__(module, {}, {}, factory).__dict__

            # install the (tag,constructor) pair in our lookup table
            record = symbols[factory]
            dtd[record.tag] = record

            if DTDBuilder.CALLBACK_CHECK == 'strict':
                # verify that there is a handler for the endElement event
                callback = 'on' + factory
                if callback not in cls.__dict__:
                    import journal
                    warning = journal.warning("pyre.xml.parsing")
                    warning.log("class '%s' should define a method 'on%s'" % (name, factory))

        cls._mydtd = dtd

        return


# version
__id__ = "$Id: DTDBuilder.py,v 1.1.1.1 2006-11-27 00:10:09 aivazis Exp $"

# End of file 
