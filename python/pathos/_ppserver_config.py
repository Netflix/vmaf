#!/usr/bin/env python
"""defalut ppserver host and port configuration"""

#tunnelports = ['12345','67890']
tunnelports = []

ppservers = tuple(["localhost:%s" % port for port in tunnelports])

# End of file
