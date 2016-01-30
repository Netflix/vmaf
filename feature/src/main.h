#pragma once

#ifndef MAIN_H_
#define MAIN_H_

int adm(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

int ansnr(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

int vif(const char *ref_path, const char *dis_path, int w, int h, const char *fmt);

int motion(const char *dis_path, int w, int h, const char *fmt);

#endif /* MAIN_H_ */
