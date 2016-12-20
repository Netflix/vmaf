clear all; close all; clc
path('D:\GSM\matlabPyrTools\',path);
im = pgmRead('einstein.pgm');
Nsc = 4; Nor = 3;
[pyr,pind] = buildSFpyr(im,Nsc,Nor);
nband = 2; 
aux = pyrBand(pyr, pind, nband);