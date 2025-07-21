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

#ifndef __VMAF_SRC_THREAD_LOCALE_H__
#define __VMAF_SRC_THREAD_LOCALE_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle for thread-local locale management.
 * Stores platform-specific locale state.
 */
typedef struct VmafThreadLocaleState VmafThreadLocaleState;

/**
 * Push "C" locale (all categories) in the current thread.
 * This ensures consistent I/O behavior regardless of system locale settings:
 * - Numeric formatting (period as decimal separator)
 * - Character classification (isalpha, isdigit, etc.)
 * - String collation and comparison
 * - Date/time formatting
 * 
 * Thread-safe: affects only the calling thread.
 * 
 * @return Opaque state handle to be passed to vmaf_thread_locale_pop(),
 *         or NULL on allocation failure or if thread-safe locale is unavailable.
 */
VmafThreadLocaleState* vmaf_thread_locale_push_c(void);

/**
 * Restore previous locale state in the current thread.
 * Must be called with the state returned by vmaf_thread_locale_push_c().
 * 
 * @param state State handle from vmaf_thread_locale_push_c().
 *              Freed internally, do not use after this call.
 */
void vmaf_thread_locale_pop(VmafThreadLocaleState* state);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_SRC_THREAD_LOCALE_H__ */
