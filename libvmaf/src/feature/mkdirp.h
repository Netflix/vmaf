
//
// mkdirp.h
//
// Copyright (c) 2013 Stephen Mathieson
// MIT licensed
//

#ifndef MKDIRP
#define MKDIRP

#include <sys/types.h>
#include <sys/stat.h>

#ifdef _MSC_VER
/* On MSVC provide a minimal mode_t typedef */
typedef int mode_t;
#endif

/*
 * Recursively `mkdir(path, mode)`
 */

int mkdirp(const char *, mode_t );

#endif
