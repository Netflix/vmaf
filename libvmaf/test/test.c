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

#include "test.h"
#include <stdio.h>

int mu_tests_run;

int main(void)
{
    char *msg = run_tests();

    if (msg)
        fprintf(stderr, "\033[31m, %s\n%d tests run, 1 failed\033[0m\n", msg, mu_tests_run);
    else
        fprintf(stderr, "\033[32m%d tests run, %d passed\033[0m\n", mu_tests_run, mu_tests_run);

    return msg != 0;
}
