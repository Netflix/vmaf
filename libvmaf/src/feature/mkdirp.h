
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

#ifdef _WIN32
typedef unsigned short mode_t;
#endif

/*
 * Recursively `mkdir(path, mode)`
 */

int mkdirp(const char *, mode_t );

#endif
