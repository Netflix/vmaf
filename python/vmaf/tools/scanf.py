"""scanf.py: scanf-style input for Python.

Danny Yoo (dyoo@hkn.eecs.berkeley.edu)

The initial motivation for this module was based on a posting on
Python-tutor:

    http://mail.python.org/pipermail/tutor/2004-July/030480.html

I haven't been able to find a nice module to do scanf-style input.
Even the Library Reference recommends regular expressions as a
substitute:

    http://docs.python.org/lib/node109.html

But there appears to have been activity about this on python-list:

    http://aspn.activestate.com/ASPN/Mail/Message/python-list/785450


Still, let's see if we can get a close equivalent scanf() in place.
At the least, it'll be fun for me, and it might be useful for people
who are still recovering from C.  *grin*


Functions provided:

    scanf(formatString) -- formatted scanning across stdin

    sscanf(sourceString, formatString) -- formated scanning across strings

    fscanf(sourceFile, formatString) -- formated scanning across files


The behavior of this scanf() will be slightly different from that
defined in C, because, in truth, I'm a little lazy, and am not quite
sure if people will need all of scanf's features in typical Python
programming.


But let's first show what conversions this scanf() will support.
Format strings are of the following form:

    % [*] [width] [format]

where [*] and [width] are optional, and [format] is mandatory.  The
optional flags modify the format.

    *         suppresses variable capture.
    width     maximum character width.


We support the following scanf conversion formats (copied from K&R):

    d    decimal integer.

    i    integer.  The integer may be in octal (leading zero) or
         hexadecimal (leading 0x or 0X).  ## fixme

    o    octal integer (with or without leading zero).  ## fixme

    x    hexadecimal integer (with or without leading 0x or 0X)   ## fixme

    c    characters.  The next input characters (default 1) are
         placed at the indicated spot.  The normal skip over white space
         is suppressed; to read the next non-white space character, use
         %1s.

    s    character string (not quoted).

    f    floating-point number with optional sign and optional decimal point.

    %    literal %; no assignment is made.


Literal characters can appear in the scanf format string: they must
match the same characters in the input.

There is no guarantee of what happens if calls to scanf are mixed with
other input functions.  See the BUGS section below for details on this.


If the input doesn't conform to the format string, a FormatError is
raised.


Example format strings:

    "%d %d"         Two decimal integers.

    "%d.%d.%d.%d"   Four decimal integers, separated by literal periods.
                    The periods won't be captured.

    "hello %s"      Literally matches "hello" followed by any number of
                    spaces, followed by a captured word.


There's also an interface for calling the internal function bscanf()
that works on CharacterBuffer types, if in the future there is
something that supports getc() and ungetc() natively.  There's also an
undocumented compile() function that takes format strings and returns
a function that can scan through CharacterBuffers.  Ooops, I guess I
just documented it.  *grin*


######################################################################


BUGS and GOTCHAS:

One major problem that I'm running into is a lack of ungetc(); it
would be nice if there were such a function in Python, but I can't
find it.  I have to simulate it by using a CharacterBuffer object, but
it's not an ideal solution.

So at most, you may lose a single character to the internal buffers
maintained by this module if you use scanf().  The other two *scanf()
functions, thankfully, aren't effected by this problem, since I can
simulate ungetc() more accurately by using seek() in the other two
cases.

If you really need to get that buffered character back, you can grab
it through _STDIN.lastChar, though manually fiddling with this is not
recommended.

So use scanf() with the following caveat: unlike C's stdin(), this
version scanf() can't be interchanged with calls to other input
functions without some kind of weird side effect.  We keep a
one-character buffer into stdin, so at most you might lose one
character to the internal buffers.

fscanf() is only allowed to work on things that support both read(1)
and seek(1, -1), since then I can reliably do a ungetch-like thing.

scanf("%s") can be dangerous in a hostile environment, since it's very
possible for something to pass in a huge string without spaces.  So use
an explicit width instead if you can help it.

######################################################################

LICENSE

Copyright (c) 2008, Danny yoo
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import string
import sys
import unittest

try:
    import StringIO

    StringIO = StringIO.StringIO  # TODO: remove this once python2 support is dropped

except ImportError:
    from io import StringIO


__all__ = ['scanf', 'sscanf', 'fscanf']
__version__ = '1.0'

# We keep a few sets as module variables just to incur the cost of constructing them just once.
_DIGIT_SET = set(string.digits)
_HEX_SET = set("0123456789ABCDEFabcdef")
_OCT_SET = set("01234567")
_PLUS_MINUS_SET = set("+-")
_WHITESPACE_SET = set(string.whitespace)


class CharacterBuffer(object):
    """A CharacterBuffer allows us to get a character, and to "unget" a
    character.  Abstract class"""

    def getch(self):
        """Returns the next character.  If there are no more characters
        left in the stream, returns the empty string."""
        pass # implement me!

    def ungetch(self, ch):
        """Tries to put back a character.  Can be called at most once
        between calls to getch()."""
        pass # implement me!

    def scanCharacterSet(self, characterSet, maxChars=0):
        """Support function that scans across a buffer till we hit
        something outside the allowable characterSet."""
        return self.scanPredicate(lambda ch: ch in characterSet, maxChars)

    def scanPredicate(self, predicate, maxChars=0):
        """Support function that scans across a buffer till we hit
        something outside what's allowable by the predicate."""
        chars = []
        countChars = 0
        while True:
            if maxChars != 0 and countChars >= maxChars:
                break
            ch = self.getch()
            if ch != '' and predicate(ch):
                chars.append(ch)
                countChars += 1
            else:
                self.ungetch(ch)
                break
        return ''.join(chars)


class CharacterBufferFromIterable(CharacterBuffer):
    """Implementation of CharacterBuffers for iterable things.
    We keep a 'lastChar' attribute to simulate ungetc()."""
    def __init__(self, iterable):
        self.iterator = iter(iterable)
        self.lastChar = ''

    def getch(self):
        if self.lastChar == '':
            try:
                return next(self.iterator)
            except StopIteration:
                return ''
        else:
            (ch, self.lastChar) = (self.lastChar, '')
            return ch

    def ungetch(self, ch):
        self.lastChar = ch


class CharacterBufferFromStdin(CharacterBuffer):
    def __init__(self, stdin):
        self.stdin = stdin
        self.lastChar = ''

    def getch(self):
        if self.lastChar == '':
            try:
                return self.stdin.read(1)
            except StopIteration:
                return ''
        else:
            (ch, self.lastChar) = (self.lastChar, '')
            return ch

    def ungetch(self, ch):
        self.lastChar = ch


class CharacterBufferFromFile(CharacterBuffer):
    """Implementation of CharacterBuffers for files.  We use the native
    read(1) and seek() calls, so we don't have to do so much magic."""
    def __init__(self, myfile):
        self.myfile = myfile

    def getch(self):
        return self.myfile.read(1)

    def ungetch(self, ch):
        self.myfile.seek(- len(ch), 1)


def readiter(inputFile, *args):
    """Returns an iterator that calls read(*args) on the inputFile."""
    while True:
        ch = inputFile.read(*args)
        if ch:
            yield ch
        else:
            raise StopIteration


def isIterable(thing):
    """Returns true if 'thing' looks iterable."""
    try:
        iter(thing)
    except TypeError:
        return False
    return True


def isFileLike(thing):
    """Returns true if thing looks like a file."""
    if hasattr(thing, "read") and hasattr(thing, "seek"):
        try:
            thing.seek(1, 1)
            thing.seek(-1, 1)
            return True
        except IOError:
            pass
    return False


def makeCharBuffer(thing):
    """Try to coerse 'thing' into a CharacterBuffer.  'thing' can be
    an instance of:

        1.  CharacterBuffer
        2.  A file-like object,
        3.  An iterable.

    makeCharBuffer() will make guesses in that order.
    """
    if isinstance(thing, CharacterBuffer):
        return thing
    elif isFileLike(thing):
        # this check must come before isIterable, since files
        # provide a line-based iterator that we don't want to use.
        # Plus we want to take advantage of file.seek()
        return CharacterBufferFromFile(thing)
    elif isIterable(thing):
        return CharacterBufferFromIterable(thing)
    else:
        raise ValueError("Can't coerse %r to CharacterBuffer" % thing)


class CappedBuffer(CharacterBuffer):
    """Implementation of a buffer that caps the number of bytes we can
    getch().  The cap may or may not include whitespace characters."""
    def __init__(self, buffer, width, ignoreWhitespace=False):
        self.buffer = buffer
        self.bytesRead = 0
        self.width = width
        self.ignoreWhitespace = ignoreWhitespace

    def getch(self):
        if self.bytesRead < self.width:
            nextChar = self.buffer.getch()
            if not self.isIgnoredChar(nextChar):
                self.bytesRead += len(nextChar)
            return nextChar
        else:
            return ''

    def isIgnoredChar(self, ch):
        return self.ignoreWhitespace and isWhitespaceChar(ch)

    def ungetch(self, ch):
        self.buffer.ungetch(ch)
        if not self.isIgnoredChar(ch):
            self.bytesRead -= len(ch)
        # make sure wacky things don't happen when ungetch()ing.
        assert self.bytesRead >= 0


class FormatError(ValueError):
    """A FormatError is raised if we run into errors while scanning
    for input."""
    pass

class IncompleteCaptureError(ValueError):
    """The *scanf() functions raise IncompleteCaptureError if a problem
    occurs doing scanning."""
    pass


"""We keep a module-level STDIN CharacterBuffer, so that we can call
scanf() several times and not lose characters between invocations."""
_STDIN = CharacterBufferFromStdin(sys.stdin)


def scanf(formatString):
    """scanf(formatString) -> tuple

Scans standard input for formats specified in the formatString.  See
module's docs for list of supported format characters."""
    return bscanf(_STDIN, formatString)


def sscanf(inputString, formatString):
    """sscanf(inputString, formatString) -> tuple

Scans inputString for formats specified in the formatString.  See
module's docs for list of supported format characters."""
    return bscanf(CharacterBufferFromIterable(inputString), formatString)


def fscanf(inputFile, formatString):
    """fscanf(inputFile, formatString) -> tuple

Scans inputFile for formats specified in the formatString.  See
module's docs for list of supported format characters."""
    buffer = CharacterBufferFromFile(inputFile)
    return bscanf(buffer, formatString)


def bscanf(buffer, formatString):
    """fscanf(buffer, formatString) -> tuple

Scans a CharacterBuffer 'buffer' for formats specified in the
formatString.  See scanf module's docs for list of supported format
characters."""
    # TODO: we may want to do some caching here of compiled formatStrings,
    # similar to that of the 're' module.
    parser = compile(formatString)
    return parser(buffer)


def isWhitespaceChar(ch, _set=_WHITESPACE_SET):
    """Returns true if the charcter looks like whitespace.
    We follow the definition of C's isspace() function.
    """
    return ch in _set


def handleWhitespace(buffer):
    """Scans for whitespace.  Returns all the whitespace it collects."""
    chars = []
    while True:
        ch = buffer.getch()
        if isWhitespaceChar(ch):
            chars.append(ch)
        else:
            buffer.ungetch(ch)
            break
    return ''.join(chars)


def handleDecimalInt(buffer, optional=False, allowLeadingWhitespace=True):
    """Tries to scan for an integer.  If 'optional' is set to False,
    returns None if an integer can't be successfully scanned."""
    if allowLeadingWhitespace:
        handleWhitespace(buffer)  # eat leading spaces
    chars = []
    chars += buffer.scanCharacterSet(_PLUS_MINUS_SET, 1)
    chars += buffer.scanCharacterSet(_DIGIT_SET)
    try:
        return int(''.join(chars), 10)
    except ValueError:
        if optional:
            return None
        raise FormatError("invalid literal characters: %s" % ''.join(chars))


def handleOct(buffer):
    chars = []
    chars += buffer.scanCharacterSet(_PLUS_MINUS_SET)
    chars += buffer.scanCharacterSet(_OCT_SET)
    try:
        return int(''.join(chars), 8)
    except ValueError:
        raise FormatError("invalid literal characters: %s" % ''.join(chars))


def handleInt(buffer, base=0):
    chars = []
    chars += buffer.scanCharacterSet(_PLUS_MINUS_SET)
    chars += buffer.scanCharacterSet("0")
    if chars and chars[-1] == '0':
        chars += buffer.scanCharacterSet("xX")
    chars += buffer.scanCharacterSet(_HEX_SET)
    try:
        return int(''.join(chars), base)
    except ValueError:
        raise FormatError("invalid literal characters: %s" % ''.join(chars))


def handleHex(buffer):
    return handleInt(buffer, 16)


def handleFloat(buffer, allowLeadingWhitespace=True):
    if allowLeadingWhitespace:
        handleWhitespace(buffer) # eat leading whitespace
    chars = []
    chars += buffer.scanCharacterSet(_PLUS_MINUS_SET)
    chars += buffer.scanCharacterSet(_DIGIT_SET)
    chars += buffer.scanCharacterSet(".")
    chars += buffer.scanCharacterSet(_DIGIT_SET)
    chars += buffer.scanCharacterSet("eE")
    chars += buffer.scanCharacterSet(_PLUS_MINUS_SET)
    chars += buffer.scanCharacterSet(_DIGIT_SET)
    try:
        return float(''.join(chars))
    except ValueError:
        raise FormatError("invalid literal characters: %s" % ''.join(chars))


def handleChars(buffer,
                allowLeadingWhitespace=False,
                isBadCharacter=lambda ch: False,
                optional=False):
    """Read as many characters are there are in the buffer."""
    if allowLeadingWhitespace:
        handleWhitespace(buffer)
    chars = []
    chars += buffer.scanPredicate(lambda ch: not isBadCharacter(ch))
    if chars:
        return ''.join(chars)
    else:
        if optional:
            return None
        raise FormatError("Empty buffer.")


def handleString(buffer, allowLeadingWhitespace=True):
    """Reading a string format is just an application of reading
    characters (skipping leading spaces, and reading up to space)."""
    return handleChars(buffer,
                       allowLeadingWhitespace=allowLeadingWhitespace,
                       isBadCharacter=isWhitespaceChar)


def makeHandleLiteral(literal):
    def f(buffer, optional=False):
        ch = buffer.getch()
        if ch == literal:
            return ch
        else:
            buffer.ungetch(ch)
            if optional:
                return None
            raise FormatError("%s != %s" % (literal, ch))
    return f


def makeWidthLimitedHandler(handler, width, ignoreWhitespace=False):
    """Constructs a Handler that caps the number of bytes that can be read
    from the byte buffer."""
    def f(buffer):
        return handler(CappedBuffer(buffer, width, ignoreWhitespace))
    return f


"""Just for kicks: handleChar is a handler for a single character."""
handleChar = makeWidthLimitedHandler(handleChars, 1, ignoreWhitespace=False)


def makeIgnoredHandler(handler):
    def f(buffer):
        handler(buffer)
        return None
    return f


class CompiledPattern:
    def __init__(self, handlers, formatString):
        self.handlers = handlers
        self.formatString = formatString

    def __call__(self, buffer):
        results = []
        try:
            for h in self.handlers:
                value = h(buffer)
                # We use None as the sentinel value that ignored handlers
                # will emit.
                if value is not None:
                    results.append(value)
            return tuple(results)
        except FormatError as e:
            raise IncompleteCaptureError(e, tuple(results))

    def __repr__(self):
        return "compile(%r)" % self.formatString


def compile(formatString):
    """Given a format string, emits a new CompiledPattern that eats
    CharacterBuffers and returns captured values as a tuple.

    If there's a failure during scanning, raises IncompleteCaptureError,
    with args being a two-tuple of the FormatError, and the results that
    were captured before the error occurred.
    """
    handlers = []
    formatBuffer = CharacterBufferFromIterable(formatString)
    while True:
        ch = formatBuffer.getch()
        if ch == '':
            break
        if isWhitespaceChar(ch):
            handleWhitespace(formatBuffer)
            handlers.append(makeIgnoredHandler(handleWhitespace))
        elif ch == '%':
            handlers.append(_compileFormat(formatBuffer))
        else:
            handlers.append(makeIgnoredHandler(makeHandleLiteral(ch)))
    return CompiledPattern(handlers, formatString)


def _compileFormat(formatBuffer):
    def readOptionalSuppression():
        f = makeHandleLiteral("*")
        return f(formatBuffer, optional=True) == "*"

    def readOptionalWidth():
        return handleDecimalInt(formatBuffer,
                                optional=True,
                                allowLeadingWhitespace=False)

    def readFormat():
        return formatBuffer.getch()  # Finally, read the format

    suppression = readOptionalSuppression()
    width = readOptionalWidth()
    formatCh = readFormat()
    handler = makeFormattedHandler(suppression, width, formatCh)
    if handler:
        return handler
    else:
        # At this point, since we couldn't figure out the format, die loudly.
        raise FormatError("Invalid format character %s" % formatCh)


_FORMAT_HANDLERS = {'d': handleDecimalInt,
                    'i': handleInt,
                    'x': handleHex,
                    'o': handleOct,
                    's': handleString,
                    'f': handleFloat,
                    '%': makeIgnoredHandler(makeHandleLiteral('%'))
                    }


def makeFormattedHandler(suppression, width, formatCh):
    """Given suppression, width, and a formatType, returns a function
    that eats a buffer and returns that thing."""
    def applySuppression(handler):
        if suppression:
            return makeIgnoredHandler(handler)
        return handler

    def applyWidth(handler):
        if width is None:
            return makeWidthLimitedHandler(handler, width,
                                           ignoreWhitespace=True)
        return handler

    # 'c' is a special case: it's the only handler that can't ignore
    # whitespace.
    if formatCh == 'c':
        if width is None:
            return applySuppression(handleChar)
        else:
            return applySuppression(
                makeWidthLimitedHandler(handleChars, width,
                                        ignoreWhitespace=False))
    if formatCh in _FORMAT_HANDLERS:
        return applySuppression(applyWidth(_FORMAT_HANDLERS[formatCh]))
    else:
        return None


######################################################################
#
# Of course we have test cases.  *grin*
#

class ScanfTests(unittest.TestCase):
    def bufferFromString(self, s):
        return CharacterBufferFromIterable(s)

    def testBufferFromString(self):
        b = self.bufferFromString("hello")
        for letter in list('hello'):
            self.assertEqual(letter, b.getch())
        self.assertEqual('', b.getch())

    def testCharacterSetScanning(self):
        b = makeCharBuffer("+++-+++++1234")
        self.assertEqual("+++", b.scanCharacterSet(set("+")))
        self.assertEqual("", b.scanCharacterSet(set("+")))
        self.assertEqual("-", b.scanCharacterSet(set("-")))
        self.assertEqual("+", b.scanCharacterSet(set("+"), 1))

    def testPredicateScanning(self):
        b = makeCharBuffer("+++-+++++1234")
        self.assertEqual("+++", b.scanPredicate(lambda ch: ch == '+'))

    def testUngetch(self):
        b = self.bufferFromString("ong")
        b.ungetch('y')
        self.assertEqual('y', b.getch())
        self.assertEqual('o', b.getch())
        b.ungetch('u')
        self.assertEqual('u', b.getch())
        self.assertEqual('n', b.getch())
        self.assertEqual('g', b.getch())
        self.assertEqual('', b.getch())

    def testRepeatedGetchOnEmptyStreamIsOk(self):
        b = self.bufferFromString("")
        self.assertEqual('', b.getch())
        self.assertEqual('', b.getch())

    def testCappedBuffer(self):
        b = CappedBuffer(self.bufferFromString("supercalifragilisticexpialidocious"), 5)
        self.assertEqual("s", b.getch())
        self.assertEqual("u", b.getch())
        self.assertEqual("p", b.getch())
        self.assertEqual("e", b.getch())
        self.assertEqual("r", b.getch())
        self.assertEqual('', b.getch())
        self.assertEqual('', b.getch())
        b.ungetch('r')
        self.assertEqual("r", b.getch())
        self.assertEqual('', b.getch())

    def testWhitespaceScanning(self):
        b = self.bufferFromString("    42\n43")
        self.assertEqual("    ", handleWhitespace(b))
        self.assertEqual("", handleWhitespace(b))
        self.assertEqual("4", b.getch())

    def testDecimalDigitScanning(self):
        b = self.bufferFromString("42 43!44")
        self.assertEqual(42, handleDecimalInt(b))
        self.assertEqual(" ", handleWhitespace(b))
        self.assertEqual(43, handleDecimalInt(b))

        b2 = self.bufferFromString("-1-2+3-4")
        self.assertEqual(-1, handleDecimalInt(b2))
        self.assertEqual(-2, handleDecimalInt(b2))
        self.assertEqual(3, handleDecimalInt(b2))
        self.assertEqual(-4, handleDecimalInt(b2))
        self.assertRaises(FormatError, handleDecimalInt, b2)

    def testCharacter(self):
        b = self.bufferFromString("hi!")
        self.assertEqual("h", handleChar(b))
        self.assertEqual("i", handleChar(b))
        self.assertEqual("!", handleChar(b))
        self.assertRaises(FormatError, handleChar, b)

    def testString(self):
        b = self.bufferFromString("-42 + 1 equals -41")
        self.assertEqual("-42", handleString(b))
        handleWhitespace(b)
        self.assertEqual("+", handleString(b))
        handleWhitespace(b)
        self.assertEqual("1", handleString(b))
        handleWhitespace(b)
        self.assertEqual("equals", handleString(b))
        handleWhitespace(b)
        self.assertEqual("-41", handleString(b))

    def testIntegerScanning(self):
        self.assertEqual((42, 43),
                          sscanf("   42\n   43  ", "%d %d"))
        self.assertEqual((8,), sscanf("10", "%o"))
        self.assertEqual((8,), sscanf("010", "%o"))
        self.assertEqual((15,), sscanf("F", "%x"))
        self.assertEqual((15,), sscanf("f", "%x"))
        self.assertEqual((15,), sscanf("0xF", "%x"))
        self.assertEqual((15,), sscanf("0XF", "%x"))
        self.assertEqual((15,), sscanf("0Xf", "%x"))
        self.assertEqual((-1, -2, 3, -4), sscanf("-1-2+3-4", "%d%d%d%d"))

    def testWordScanning(self):
        self.assertEqual(("hello", "world"),
                          sscanf("   hello world", "%s %s"))

    def testSuppression(self):
        self.assertEqual((), sscanf(" hello world", "%*s %*s"))
        self.assertEqual(("happy",),
                          sscanf("hello happy world", "%*s %s %*s"))
        self.assertEqual((), sscanf("h", "%*c"))

    def testWidth(self):
        self.assertEqual(("00010",), sscanf("00010101010111", "%5c"))
        self.assertEqual(("xy",), sscanf("xyz", "%2s"))
        self.assertEqual(("xy",), sscanf("              xyz", "%2s"))
        self.assertEqual(("  ",), sscanf("              xyz", "%2c"))

    def testFscanf(self):
        b = StringIO("hello world")
        self.assertEqual(("hello", " ", "world"), fscanf(b, "%s%c%s"))
        # Check that calling fscanf() twice doesn't
        # drop the last character
        b2 = StringIO("hello world")
        self.assertEqual(("hello",), fscanf(b2, "%s"))
        self.assertEqual((" ",), fscanf(b2, "%c"))
        self.assertEqual(("world",), fscanf(b2, "%s"))

    def testSkipLeadingSpaceOnScanning(self):
        """Ralph Heinkel reported a bug where floats weren't being
        parsed properly if there were leading whitespace for %f.
        This case checks that"""
        self.assertEqual((42.0,),
                          sscanf("    42.0", "%f"))

    def testFloats(self):
        self.assertEqual((3.14,
                           10.,
                           .001,
                           1e100,
                           3.14e-10,
                           0e0,), sscanf("""3.14
                           10.
                           .001
                           1e100
                           3.14e-10
                           0e0""", "%f %f %f %f %f %f"))

    def testMoreSimpleScanningExamples(self):
        self.assertEqual((192,168,1,1),
                          sscanf("192.168.1.1", "%d.%d.%d.%d"))
        self.assertEqual(("a", "b", "c"),
                          sscanf("  ab   c  ", "%1s%1s%s"))
        self.assertEqual(("hello", " ", "world"),
                          sscanf("hello world", "%s%c%s"))
        self.assertRaises(IncompleteCaptureError,
                          sscanf, "192.168.1.1", "%d %d %d %d")
        self.assertEqual(("danny",),
                          sscanf("hi danny", "hi %s"))
        self.assertEqual(("danny",),
                          sscanf("  hi danny", "  hi %s"))
        self.assertEqual(("a", "b", 3),
                          sscanf("ab3", "%c%c%d"))
        # this case is weird, but it happens in C too!
        self.assertRaises(IncompleteCaptureError,
                          sscanf, "  hi danny", "hi %s")

        # The example that's used in http://docs.python.org/lib/node109.html
        self.assertEqual(("/usr/bin/sendmail", 0, 4),
                          sscanf("/usr/bin/sendmail - 0 errors, 4 warnings",
                                 "%s - %d errors, %d warnings"))

    def testErroneousFormats(self):
        self.assertRaises(FormatError, compile, "%")
        self.assertRaises(FormatError, compile, "% ")
        self.assertRaises(FormatError, compile, "%*")
        self.assertRaises(FormatError, compile, "%*z")
        self.assertRaises(FormatError, compile, "% d")
        self.assertRaises(FormatError, compile, "%* d")


if __name__ == '__main__':
    unittest.main()
