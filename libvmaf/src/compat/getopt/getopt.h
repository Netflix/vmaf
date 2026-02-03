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
#ifndef YA_GETOPT_H
#define YA_GETOPT_H 1

#if defined(__cplusplus)
extern "C" {
#endif

#define ya_no_argument        0
#define ya_required_argument  1
#define ya_optional_argument  2

struct option {
    const char *name;
    int has_arg;
    int *flag;
    int val;
};

int ya_getopt(int argc, char * const argv[], const char *optstring);
int ya_getopt_long(int argc, char * const argv[], const char *optstring,
                   const struct option *longopts, int *longindex);
int ya_getopt_long_only(int argc, char * const argv[], const char *optstring,
                        const struct option *longopts, int *longindex);

extern char *ya_optarg;
extern int ya_optind, ya_opterr, ya_optopt;

#ifndef YA_GETOPT_NO_COMPAT_MACRO
#define getopt ya_getopt
#define getopt_long ya_getopt_long
#define getopt_long_only ya_getopt_long_only
#define optarg ya_optarg
#define optind ya_optind
#define opterr ya_opterr
#define optopt ya_optopt
#define no_argument ya_no_argument
#define required_argument ya_required_argument
#define optional_argument ya_optional_argument
#endif

#if defined(__cplusplus)
}
#endif

#endif
