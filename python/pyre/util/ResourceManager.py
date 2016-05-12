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


from Resource import Resource


class ResourceManager(Resource):


    def find(self, name):
        return self._registry.get(name)


    def manage(self, resource, id, aliases=[]):
        self._resources[resource] = tuple([id] + aliases)
        self._registry[id] = resource
        for alias in aliases:
            self._registry[alias] = resource
            
        return


    def resources(self):
        return self._resources.keys()


    def __init__(self, name):
        Resource.__init__(self, name)

        self._registry = {}
        self._resources = {}

        return


# version
__id__ = "$Id: ResourceManager.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
