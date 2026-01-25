
//
// mkdirp.c
//
// Copyright (c) 2013 Stephen Mathieson
// MIT licensed
//

#ifdef _MSC_VER
# include <io.h>
# include <direct.h> /* _mkdir, _wmkdir */
# ifndef strdup
#  define strdup _strdup
# endif
#else
# include <unistd.h>
#endif
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "mkdirp.h"

static char *
path_normalize(const char *path) {
  if (!path) return NULL;

  char *copy = strdup(path);
  if (NULL == copy) return NULL;
  char *ptr = copy;

  for (int i = 0; copy[i]; i++) {
    *ptr++ = path[i];
    if ('/' == path[i]) {
      i++;
      while ('/' == path[i]) i++;
      i--;
    }
  }

  *ptr = '\0';

  return copy;
}

#ifdef _WIN32
#define PATH_SEPARATOR   '\\'
#else
#define PATH_SEPARATOR   '/'
#endif

int
mkdirp(const char *path, mode_t mode) {
  char *pathname = NULL;
  char *parent = NULL;

  if (NULL == path) return -1;

  pathname = path_normalize(path);
  if (NULL == pathname) goto fail;

  parent = strdup(pathname);
  if (NULL == parent) goto fail;

  char *p = parent + strlen(parent);
  while (PATH_SEPARATOR != *p && p != parent) {
    p--;
  }
  *p = '\0';

  // make parent dir
  if (p != parent && 0 != mkdirp(parent, mode)) goto fail;
  free(parent);

  // make this one if parent has been made
#ifdef _MSC_VER
  /* On MSVC use _mkdir which takes only a path.
   * Microsoft docs: _mkdir and _wmkdir create a new directory and return 0
   * on success or -1 on error, setting errno accordingly. See CRT docs:
   * https://learn.microsoft.com/en-us/c-runtime-library/reference/mkdir-wmkdir
   *
   * The CRT documents that _mkdir/_wmkdir behave like mkdir but accept only
   * a path (no mode) and set errno on failure (EEXIST, ENOENT, ...). */
  int rc = _mkdir(pathname);
#else
  int rc = mkdir(pathname, mode);
#endif

  free(pathname);

  return 0 == rc || EEXIST == errno
    ? 0
    : -1;

fail:
  free(pathname);
  free(parent);
  return -1;
}
