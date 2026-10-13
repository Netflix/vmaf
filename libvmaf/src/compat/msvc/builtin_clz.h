#pragma once

static inline int __builtin_clz(unsigned x) {
    if (x == 0) return 32;
    unsigned n = 0;
    if ((x & 0xFFFF0000) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF000000) == 0) { n += 8;  x <<= 8; }
    if ((x & 0xF0000000) == 0) { n += 4;  x <<= 4; }
    if ((x & 0xC0000000) == 0) { n += 2;  x <<= 2; }
    if ((x & 0x80000000) == 0) { n += 1; }
    return n;
}

static inline int __builtin_clzll(unsigned long long x) {
    if (x == 0) return 64;
    unsigned n = 0;
    if ((x & 0xFFFFFFFF00000000ULL) == 0) { n += 32; x <<= 32; }
    if ((x & 0xFFFF000000000000ULL) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF00000000000000ULL) == 0) { n += 8;  x <<= 8; }
    if ((x & 0xF000000000000000ULL) == 0) { n += 4;  x <<= 4; }
    if ((x & 0xC000000000000000ULL) == 0) { n += 2;  x <<= 2; }
    if ((x & 0x8000000000000000ULL) == 0) { n += 1; }
    return n;
}
