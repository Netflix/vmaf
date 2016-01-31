/**
 *
 *  Copyright 2016 Netflix, Inc.
 *
 *     Licensed under the GNU Lesser General Public License, Version 3
 *     (the "License"); you may not use this file except in compliance
 *     with the License. You may obtain a copy of the License at
 *
 *         http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 *     implied. See the License for the specific language governing
 *     permissions and limitations under the License.
 *
 */

#ifndef ALIGNMENT_H_
#define ALIGNMENT_H_

// int vmaf_floorn(int n, int m) // O0
inline int vmaf_floorn(int n, int m) // O1, O2, O3
{
	return n - n % m;
}

// int vmaf_ceiln(int n, int m) // O0
inline int vmaf_ceiln(int n, int m) // O1, O2, O3
{
	return n % m ? n + (m - n % m) : n;
}

#endif // ALIGNMENT_H_
