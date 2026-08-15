ya_getopt - Yet another getopt
==============================

What is ya_getopt.
------------------

Ya_getopt is a drop-in replacement of [GNU C library getopt](http://man7.org/linux/man-pages/man3/getopt.3.html).
`getopt()`, `getopt_long()` and `getopt_long_only()` are implemented excluding the following GNU extension features.

1. If *optstring* contains **W** followed by a semicolon, then **-W** **foo** is
   treated as the long option **--foo**.

2. \_\<PID>\_GNU\_nonoption\_argv\_flags\_

The license is 2-clause BSD-style license. You can use the Linux getopt compatible function
under Windows, Solaris and so on without having to worry about license issue.

Note for contributors
---------------------

Don't send me a patch if you have looked at GNU C library getopt source code.
That's because I made this with clean room design to avoid the influence of the GNU LGPL.

Please make a test script passed by the GNU C library getopt but not by ya_getopt instead.

License
-------

2-clause BSD-style license

Other getopt functions
----------------------

* [public domain AT&T getopt](https://www.google.co.jp/search?q=public+domain+at%26t+getopt) public domain, no getopt_long, no getopt_long_only, no argv permutation
* [Free Getopt](http://freegetopt.sourceforge.net/) 3-clause BSD-style licence, no getopt_long, no getopt_long_only
* [getopt_port](https://github.com/kimgr/getopt_port/) 3-clause BSD-style licence, no getopt_long_only, no argv permutation
* [XGetopt - A Unix-compatible getopt() for MFC and Win32](http://www.codeproject.com/Articles/1940/XGetopt-A-Unix-compatible-getopt-for-MFC-and-Win32)
* [Full getopt Port for Unicode and Multibyte Microsoft Visual C, C++, or MFC Projects](http://www.codeproject.com/Articles/157001/Full-getopt-Port-for-Unicode-and-Multibyte-Microso) LGPL
