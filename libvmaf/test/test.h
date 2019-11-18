#include <stdio.h>

// http://www.jera.com/techinfo/jtns/jtn002.html

#define mu_assert(message, test) \
    do {                         \
        if (!(test))             \
            return message;      \
    } while (0)

#define mu_run_test(test)                             \
    do {                                              \
        fprintf(stderr, #test ": ");                  \
        char *message = test();                       \
        mu_tests_run++;                               \
        if (message) {                                \
            fprintf(stderr, "\033[31mfail\033[0m");   \
            return message;                           \
        } else {                                      \
            fprintf(stderr, "\033[32mpass\033[0m\n"); \
        }                                             \
    } while (0)

extern int mu_tests_run;
char *run_tests(void);
