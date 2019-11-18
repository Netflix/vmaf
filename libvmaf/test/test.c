#include "test.h"
#include <stdio.h>

int mu_tests_run;

int main(int argc, char *argv[])
{
    char *msg = run_tests();

    if (msg)
        fprintf(stderr, "\033[31m, %s\n%d tests run, 1 failed\033[0m\n", msg, mu_tests_run);
    else
        fprintf(stderr, "\033[32m%d tests run, %d passed\033[0m\n", mu_tests_run, mu_tests_run);

    return msg != 0;
}
