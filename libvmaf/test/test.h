/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

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
