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

#pragma once

#ifndef MAIN_H_
#define MAIN_H_

int adm(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

int ansnr(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

int vif(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

int motion(const char *dis_path, int w, int h, const char *fmt);

#endif /* MAIN_H_ */
