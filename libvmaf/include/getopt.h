#ifndef VMAF_GETOPT_H
#define VMAF_GETOPT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern char *optarg;
extern int optind;
extern int opterr;
extern int optopt;

int getopt(int argc, char *const *argv, const char *optstring);
int getopt_long(int argc, char *const *argv, const char *shortopts,
                const struct option *longopts, int32_t *longind);
int getopt_long_only(int argc, char *const *argv, const char *shortopts,
                     const struct option *longopts, int32_t *longind);

struct option {
    const char *name;
    int has_arg;
    int *flag;
    int val;
};

#define no_argument 0
#define required_argument 1
#define optional_argument 2

#ifdef __cplusplus
}
#endif

#endif /* VMAF_GETOPT_H */
