/* -*- indent-tabs-mode: nil -*-
 *
 * ya_getopt  - Yet another getopt
 * https://github.com/kubo/ya_getopt
 *
 * Copyright 2015 Kubo Takehiro <kubo@jiubao.org>
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *    1. Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright notice, this list
 *       of conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those of the
 * authors and should not be interpreted as representing official policies, either expressed
 * or implied, of the authors.
 *
 */
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "getopt.h"

char *ya_optarg = NULL;
int ya_optind = 1;
int ya_opterr = 1;
int ya_optopt = '?';
static char *ya_optnext = NULL;
static int posixly_correct = -1;
static int handle_nonopt_argv = 0;

static void ya_getopt_error(const char *optstring, const char *format, ...);
static void check_gnu_extension(const char *optstring);
static int ya_getopt_internal(int argc, char * const argv[], const char *optstring, const struct option *longopts, int *longindex, int long_only);
static int ya_getopt_shortopts(int argc, char * const argv[], const char *optstring, int long_only);
static int ya_getopt_longopts(int argc, char * const argv[], char *arg, const char *optstring, const struct option *longopts, int *longindex, int *long_only_flag);

static void ya_getopt_error(const char *optstring, const char *format, ...)
{
    if (ya_opterr && optstring[0] != ':') {
        va_list ap;
        va_start(ap, format);
        vfprintf(stderr, format, ap);
        va_end(ap);
    }
}

static void check_gnu_extension(const char *optstring)
{
    if (optstring[0] == '+' || getenv("POSIXLY_CORRECT") != NULL) {
        posixly_correct = 1;
    } else {
        posixly_correct = 0;
    }
    if (optstring[0] == '-') {
        handle_nonopt_argv = 1;
    } else {
        handle_nonopt_argv = 0;
    }
}

static int is_option(const char *arg)
{
    return arg[0] == '-' && arg[1] != '\0';
}

int ya_getopt(int argc, char * const argv[], const char *optstring)
{
    return ya_getopt_internal(argc, argv, optstring, NULL, NULL, 0);
}

int ya_getopt_long(int argc, char * const argv[], const char *optstring, const struct option *longopts, int *longindex)
{
    return ya_getopt_internal(argc, argv, optstring, longopts, longindex, 0);
}

int ya_getopt_long_only(int argc, char * const argv[], const char *optstring, const struct option *longopts, int *longindex)
{
    return ya_getopt_internal(argc, argv, optstring, longopts, longindex, 1);
}

static int ya_getopt_internal(int argc, char * const argv[], const char *optstring, const struct option *longopts, int *longindex, int long_only)
{
    static int start, end;

    if (ya_optopt == '?') {
        ya_optopt = 0;
    }

    if (posixly_correct == -1) {
        check_gnu_extension(optstring);
    }

    if (ya_optind == 0) {
        check_gnu_extension(optstring);
        ya_optind = 1;
        ya_optnext = NULL;
    }

    switch (optstring[0]) {
    case '+':
    case '-':
        optstring++;
    }

    if (ya_optnext == NULL && start != 0) {
        int last_pos = ya_optind - 1;

        ya_optind -= end - start;
        if (ya_optind <= 0) {
            ya_optind = 1;
        }
        while (start < end--) {
            int i;
            char *arg = argv[end];

            for (i = end; i < last_pos; i++) {
                ((char **)argv)[i] = argv[i + 1];
            }
            ((char const **)argv)[i] = arg;
            last_pos--;
        }
        start = 0;
    }

    if (ya_optind >= argc) {
        ya_optarg = NULL;
        return -1;
    }
    if (ya_optnext == NULL) {
        const char *arg = argv[ya_optind];
        if (!is_option(arg)) {
            if (handle_nonopt_argv) {
                ya_optarg = argv[ya_optind++];
                start = 0;
                return 1;
            } else if (posixly_correct) {
                ya_optarg = NULL;
                return -1;
            } else {
                int i;

                start = ya_optind;
                for (i = ya_optind + 1; i < argc; i++) {
                    if (is_option(argv[i])) {
                        end = i;
                        break;
                    }
                }
                if (i == argc) {
                    ya_optarg = NULL;
                    return -1;
                }
                ya_optind = i;
                arg = argv[ya_optind];
            }
        }
        if (strcmp(arg, "--") == 0) {
            ya_optind++;
            return -1;
        }
        if (longopts != NULL && arg[1] == '-') {
            return ya_getopt_longopts(argc, argv, argv[ya_optind] + 2, optstring, longopts, longindex, NULL);
        }
    }

    if (ya_optnext == NULL) {
        ya_optnext = argv[ya_optind] + 1;
    }
    if (long_only) {
        int long_only_flag = 0;
        int rv = ya_getopt_longopts(argc, argv, ya_optnext, optstring, longopts, longindex, &long_only_flag);
        if (!long_only_flag) {
            ya_optnext = NULL;
            return rv;
        }
    }

    return ya_getopt_shortopts(argc, argv, optstring, long_only);
}

static int ya_getopt_shortopts(int argc, char * const argv[], const char *optstring, int long_only)
{
    int opt = *ya_optnext;
    const char *os = strchr(optstring, opt);

    if (os == NULL) {
        ya_optarg = NULL;
        if (long_only) {
            ya_getopt_error(optstring, "%s: unrecognized option '-%s'\n", argv[0], ya_optnext);
            ya_optind++;
            ya_optnext = NULL;
        } else {
            ya_optopt = opt;
            ya_getopt_error(optstring, "%s: invalid option -- '%c'\n", argv[0], opt);
            if (*(++ya_optnext) == 0) {
                ya_optind++;
                ya_optnext = NULL;
            }
        }
        return '?';
    }
    if (os[1] == ':') {
        if (ya_optnext[1] == 0) {
            ya_optind++;
            ya_optnext = NULL;
            if (os[2] == ':') {
                /* optional argument */
                ya_optarg = NULL;
            } else {
                if (ya_optind == argc) {
                    ya_optarg = NULL;
                    ya_optopt = opt;
                    ya_getopt_error(optstring, "%s: option requires an argument -- '%c'\n", argv[0], opt);
                    if (optstring[0] == ':') {
                        return ':';
                    } else {
                        return '?';
                    }
                }
                ya_optarg = argv[ya_optind];
                ya_optind++;
            }
        } else {
            ya_optarg = ya_optnext + 1;
            ya_optind++;
        }
        ya_optnext = NULL;
    } else {
        ya_optarg = NULL;
        if (ya_optnext[1] == 0) {
            ya_optnext = NULL;
            ya_optind++;
        } else {
            ya_optnext++;
        }
    }
    return opt;
}

static int ya_getopt_longopts(int argc, char * const argv[], char *arg, const char *optstring, const struct option *longopts, int *longindex, int *long_only_flag)
{
    char *val = NULL;
    const struct option *opt;
    size_t namelen;
    int idx;

    for (idx = 0; longopts[idx].name != NULL; idx++) {
        opt = &longopts[idx];
        namelen = strlen(opt->name);
        if (strncmp(arg, opt->name, namelen) == 0) {
            switch (arg[namelen]) {
            case '\0':
                switch (opt->has_arg) {
                case ya_required_argument:
                    ya_optind++;
                    if (ya_optind == argc) {
                        ya_optarg = NULL;
                        ya_optopt = opt->val;
                        ya_getopt_error(optstring, "%s: option '--%s' requires an argument\n", argv[0], opt->name);
                        if (optstring[0] == ':') {
                            return ':';
                        } else {
                            return '?';
                        }
                    }
                    val = argv[ya_optind];
                    break;
                }
                goto found;
            case '=':
                if (opt->has_arg == ya_no_argument) {
                    const char *hyphens = (argv[ya_optind][1] == '-') ? "--" : "-";

                    ya_optind++;
                    ya_optarg = NULL;
                    ya_optopt = opt->val;
                    ya_getopt_error(optstring, "%s: option '%s%s' doesn't allow an argument\n", argv[0], hyphens, opt->name);
                    return '?';
                }
                val = arg + namelen + 1;
                goto found;
            }
        }
    }
    if (long_only_flag) {
        *long_only_flag = 1;
    } else {
        ya_getopt_error(optstring, "%s: unrecognized option '%s'\n", argv[0], argv[ya_optind]);
        ya_optind++;
    }
    return '?';
found:
    ya_optarg = val;
    ya_optind++;
    if (opt->flag) {
        *opt->flag = opt->val;
    }
    if (longindex) {
        *longindex = idx;
    }
    return opt->flag ? 0 : opt->val;
}
