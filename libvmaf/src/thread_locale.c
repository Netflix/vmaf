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

#include "thread_locale.h"
#include "config.h"

#include <stdlib.h>
#include <string.h>

#ifdef HAVE_XLOCALE_H
#include <xlocale.h>
#endif

#include <locale.h>

// Platform-specific locale state
struct VmafThreadLocaleState {
#if defined(HAVE_USELOCALE)
    // POSIX.1-2008: thread-local locale
    locale_t c_locale;
    locale_t old_locale;
#elif defined(_WIN32)
    // Windows: thread-local locale via _configthreadlocale
    int old_per_thread_mode;
    char old_locale[256];
#else
    // No thread-safe locale support available
    void *reserved; // Reserved for future use
#endif
};

VmafThreadLocaleState* vmaf_thread_locale_push_c(void)
{
    VmafThreadLocaleState* state = malloc(sizeof(VmafThreadLocaleState));
    if (!state) return NULL;

    memset(state, 0, sizeof(VmafThreadLocaleState));

#if defined(HAVE_USELOCALE)
    // POSIX.1-2008: thread-local locale (Linux, macOS, BSD)
    // Use LC_ALL_MASK for complete locale isolation
    state->c_locale = newlocale(LC_ALL_MASK, "C", NULL);
    if (state->c_locale == (locale_t)0) {
        free(state);
        return NULL;
    }
    state->old_locale = uselocale(state->c_locale);
    
#elif defined(_WIN32)
    // Windows: enable per-thread locale, then set to "C"
    // Use LC_ALL for complete locale isolation
    state->old_per_thread_mode = _configthreadlocale(_ENABLE_PER_THREAD_LOCALE);
    if (state->old_per_thread_mode == -1) {
        free(state);
        return NULL;
    }
    
    const char* old = setlocale(LC_ALL, NULL);
    if (old) {
        strncpy(state->old_locale, old, sizeof(state->old_locale) - 1);
        state->old_locale[sizeof(state->old_locale) - 1] = '\0';
    }
    
    setlocale(LC_ALL, "C");
    
#else
    // No thread-safe locale support available on this platform
    free(state);
    return NULL;
#endif

    return state;
}

void vmaf_thread_locale_pop(VmafThreadLocaleState* state)
{
    if (!state) return;

#if defined(HAVE_USELOCALE)
    // POSIX.1-2008: restore thread-local locale
    if (state->c_locale != (locale_t)0) {
        uselocale(state->old_locale);
        freelocale(state->c_locale);
    }
    
#elif defined(_WIN32)
    // Windows: restore locale and per-thread mode
    if (state->old_locale[0] != '\0') {
        setlocale(LC_ALL, state->old_locale);
    }
    
    if (state->old_per_thread_mode != -1) {
        _configthreadlocale(state->old_per_thread_mode);
    }
#endif

    free(state);
}
