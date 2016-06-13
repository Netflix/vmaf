/**
 *
 *  Copyright 2016 Netflix, Inc.
 *
 *     Licensed under the Apache License, Version 2.0 (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#pragma once

#ifndef FEATURE_H_
#define FEATURE_H_

int adm(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);
int ansnr(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);
int vif(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);
int motion(const char *dis_path, int w, int h, const char *fmt);
int all(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

int psnr(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);
int moment(const char *path, int w, int h, const char *fmt, int order);
int ssim(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);
int ms_ssim(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

#endif /* FEATURE_H_ */
