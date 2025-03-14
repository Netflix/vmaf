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

#ifndef MACROS_H_
#define MACROS_H_

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#define UNUSED_FUNCTION /**/
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#define UNUSED_FUNCTION __attribute__((unused))
#endif
#define RESTRICT __restrict

#endif // MACROS_H_
