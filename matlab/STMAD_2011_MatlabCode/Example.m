clc;
clear;
close all;

Hei = 324;
Wid = 576;
OrgFile = '../../python/test/resource/yuv/src01_hrc00_576x324.yuv';
DstFile = '../../python/test/resource/yuv/src01_hrc01_576x324.yuv';

run_stmad(OrgFile, DstFile, Wid, Hei);
