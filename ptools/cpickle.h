#ifndef CPICKLE_H_

// This is stolen directly out of the cPickle.c Python Module (v 2.5) 

/*
 * Pickle opcodes.  These must be kept in synch with pickle.py.  Extensive
 * docs are in pickletools.py.
 */
#define PY_MARK        '('
#define PY_STOP        '.'
#define PY_POP         '0'
#define PY_POP_MARK    '1'
#define PY_DUP         '2'
#define PY_FLOAT       'F'
#define PY_BINFLOAT    'G'
#define PY_INT         'I'
#define PY_BININT      'J'
#define PY_BININT1     'K'
#define PY_LONG        'L'
#define PY_BININT2     'M'
#define PY_NONE        'N'
#define PY_PERSID      'P'
#define PY_BINPERSID   'Q'
#define PY_REDUCE      'R'
#define PY_STRING      'S'
#define PY_BINSTRING   'T'
#define PY_SHORT_BINSTRING 'U'
#define PY_UNICODE     'V'
#define PY_BINUNICODE  'X'
#define PY_APPEND      'a'
#define PY_BUILD       'b'
#define PY_GLOBAL      'c'
#define PY_DICT        'd'
#define PY_EMPTY_DICT  '}'
#define PY_APPENDS     'e'
#define PY_GET         'g'
#define PY_BINGET      'h'
#define PY_INST        'i'
#define PY_LONG_BINGET 'j'
#define PY_LIST        'l'
#define PY_EMPTY_LIST  ']'
#define PY_OBJ         'o'
#define PY_PUT         'p'
#define PY_BINPUT      'q'
#define PY_LONG_BINPUT 'r'
#define PY_SETITEM     's'
#define PY_TUPLE       't'
#define PY_EMPTY_TUPLE ')'
#define PY_SETITEMS    'u'
/* Protocol 2. */
#define PY_PROTO    '\x80' /* identify pickle protocol */
#define PY_NEWOBJ   '\x81' /* build object by applying cls.__new__ to argtuple */
#define PY_EXT1     '\x82' /* push object from extension registry; 1-byte index */
#define PY_EXT2     '\x83' /* ditto, but 2-byte index */
#define PY_EXT4     '\x84' /* ditto, but 4-byte index */
#define PY_TUPLE1   '\x85' /* build 1-tuple from stack top */
#define PY_TUPLE2   '\x86' /* build 2-tuple from two topmost stack items */
#define PY_TUPLE3   '\x87' /* build 3-tuple from three topmost stack items */
#define PY_NEWTRUE  '\x88' /* push True */
#define PY_NEWFALSE '\x89' /* push False */
#define PY_LONG1    '\x8a' /* push long from < 256 bytes */
#define PY_LONG4    '\x8b' /* push really big long */


#define CPICKLE_H_
#endif // CPICKLE_H_
