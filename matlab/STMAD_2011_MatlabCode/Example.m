clc;
clear;
close all;

% Wid = 176;
% Hei = 144;

% OrgFile = 'foreman_org_qcif.yuv';

% DstFile = 'foreman_dst_qcif.yuv';

% MadVals = STMAD_index(OrgFile, DstFile, Wid, Hei);
% MadVals = run_stmad(OrgFile, DstFile, Wid, Hei);

Hei = 324;
Wid = 576;
OrgFile = ...
    '/home/cbampis/Projects/stash/MCE/vmaf_oss/vmaf/python/test/resource/yuv/src01_hrc00_576x324.yuv';
DstFile = ...
    '/home/cbampis/Projects/stash/MCE/vmaf_oss/vmaf/python/test/resource/yuv/src01_hrc01_576x324.yuv';

run_stmad(OrgFile, DstFile, Wid, Hei);

