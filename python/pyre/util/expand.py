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


def expandMacros(text, substitutions):
    """scan <text> looking for ${macro} and expand them with text from <substitutions>"""

    global regexp
    if not regexp:
        import re
        regexp = re.compile("\$\{(.+?)\}")

    result = []
    cursor = 0
    for match in regexp.finditer(text):
        start, end = match.start(), match.end()
        try:
            replacement = substitutions[match.group(1)]
        except KeyError:
            result.append(text[cursor:end])
            cursor = end
            continue

        result.append(text[cursor:start])
        result.append(replacement)
        cursor = end
        
    result.append(text[cursor:])

    return ''.join(result)


regexp = None
    

# version
__id__ = "$Id: expand.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
