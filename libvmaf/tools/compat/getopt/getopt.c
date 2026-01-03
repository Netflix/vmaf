/* Minimal getopt compatibility implementation for platforms without getopt.h */

#include "getopt.h"
#include <stdlib.h>
#include <string.h>

char *optarg = NULL;
int optind = 1;
int opterr = 1;
int optopt = '?';

int getopt(int argc, char *const *argv, const char *optstring) {
    static int sp = 1;
    if (optind >= argc) return -1;
    char *arg = argv[optind];
    if (arg[0] != '-' || arg[1] == '\0') return -1;
    if (strcmp(arg, "--") == 0) { optind++; return -1; }
    char c = arg[sp];
    const char *oli = strchr(optstring, c);
    if (!oli) {
        optopt = c;
        if (arg[++sp] == '\0') { optind++; sp = 1; }
        return '?';
    }
    if (oli[1] == ':') {
        if (arg[sp+1] != '\0') {
            optarg = &arg[sp+1];
            optind++;
        } else if (optind+1 < argc) {
            optarg = argv[++optind];
            optind++;
        } else {
            optopt = c;
            if (optstring[0] == ':') return ':'; else return '?';
        }
        sp = 1;
    } else {
        if (arg[++sp] == '\0') { optind++; sp = 1; }
    }
    return c;
}

int getopt_long(int argc, char *const *argv, const char *shortopts,
                const struct option *longopts, int32_t *longind) {
    (void)longopts; (void)longind;
    return getopt(argc, argv, shortopts);
}

int getopt_long_only(int argc, char *const *argv, const char *shortopts,
                     const struct option *longopts, int32_t *longind) {
    return getopt_long(argc, argv, shortopts, longopts, longind);
}
