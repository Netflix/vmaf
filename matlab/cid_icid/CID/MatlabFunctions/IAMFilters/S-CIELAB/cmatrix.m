function result = cmatrix(matrixtype, spacetype)
% result = cmatrix(matrixtype, spacetype)
% 
% Returns a 3x3 color matrix used by changeColorSpace.
%
% matrixtype has the following options:
%    'lms2opp' -- cone coordinate to opponent (Poirson & Wandell 1993)
%    'opp2lms' -- inverse of the above matrix
%    'xyz2opp' -- xyz to opponent (CIE1931 2 degree XYZ)
%    'opp2xyz' -- inverse of the above matrix
%    'lms2xyz' -- Hunt-Pointer-Estevez transformation from cone
%                 to XYZ, normalized for D65 (lms=[100 100 100] for D65).
%    'xyz2lms' -- inverse of lms2xyz.
%    'xyz2yiq' -- convert from XYZ to YIQ
%    'yiq2xyz' -- inverse of the abvoe matrix
%    'rgb2yuv' -- convert from RGB to YUV (YCbCr) for JPEG compression
%    'yuv2rgb' -- inverse of the above matrix
%    'xyz2srgb' -- from XYZ to sRGB values
%    'srgb2xyz' -- inverse of the above matrix
%             (the above are not dependent on device calibration)
%
%    'rgb2lms' -- monitor rgb to cone coordinate
%    'lms2rgb' -- inverse of the above matrix
%    'rgb2xyz' -- rgb to xyz 2 degree.
%    'xyz2rgb' -- inverse of the above matrix
%             (The above transformation matrices are based on the
%              calibration of the monitor used in Poirson&Wandell(1993) 
%              experiment. If possible, you should provide this 
%              matrix based on calibration data of your own device,
%              instead of using this one. This is only a default when you
%              don't have calibration data available for your device).
%
% spacetype specifies what type of xyz space (CIE1931 2 degree or 
% CIE1964 10 degree) is required.
%    spacetype = 2: cie1931 2 degree XYZ   (default)
%    spacetype = 10: cie1964 10 degree XYZ
%   
% Xuemei Zhang 3/11/96
% Last modified 1/22/98


if (nargin < 2)
  spacetype = 2;
end

matrixtype = matrixtype(1:7);

if (matrixtype == 'lms2opp')
  result = [0.9900   -0.1060   -0.0940; ...
           -0.6690    0.7420   -0.0270; ...
           -0.2120   -0.3540    0.9110];
end

if (matrixtype == 'opp2lms')
  result = inv([0.9900   -0.1060   -0.0940; ...
               -0.6690    0.7420   -0.0270; ...
               -0.2120   -0.3540    0.9110]);
end

if (matrixtype == 'lms2xyz')
  result = inv([.4002  .7076 -.0808; ...
               -.2263 1.1653  .0457; ...
                .0     .0     .9182]);
end

if (matrixtype == 'xyz2lms')
  result = [0.4002    0.7076   -0.0808; ...
           -0.2263    1.1653    0.0457; ...
            0         0         0.9182];
end

if (matrixtype == 'xyz2opp' | matrixtype == 'opp2xyz')
  if (spacetype == 2)
    result = [278.7336  721.8031 -106.5520; ...
             -448.7736  289.8056   77.1569; ...
               85.9513 -589.9859  501.1089]/1000;
  end
  if (spacetype == 10)
    result = [ 288.5613  659.7617 -130.5654; ...
              -464.8864  326.2702   62.4200; ...
                79.8787 -554.7976  481.4746]/1000;
  end
  if (matrixtype == 'opp2xyz')
    result = inv(result);
  end
end

if (matrixtype == 'xyz2yiq' | matrixtype == 'yiq2xyz')
  result = [     0    1.0000         0; ...
            1.4070   -0.8420   -0.4510; ...
            0.9320   -1.1890    0.2330];
  if (matrixtype == 'yiq2xyz')
    result = inv(result);
  end
end

if (matrixtype == 'rgb2yuv' | matrixtype == 'yuv2rgb')
  result = [ 0.299   0.587   0.114; ...
            -0.1687 -0.3313  0.5; ...
	     0.5    -0.4187 -0.0813];
  if (matrixtype == 'yuv2rgb')
    result = inv(result);
  end
end

if (matrixtype == 'xyz2srg' | matrixtype == 'srgb2xy')
  result = [.03241 -.015374 -0.004986; ...
           -0.009692 .018760 0.000416; ...
	   0.000556 -0.002040 .01057];
  if (matrixtype == 'srgb2xy')
    result = inv(result);
  end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (matrixtype == 'rgb2lms')
  result = [12.2430   44.4548    6.5701; ...
             4.6321   44.6748    9.5109; ...
             0.5227    4.6900   44.8061];
end

if (matrixtype == 'lms2rgb')
  result = inv([12.2430   44.4548    6.5701; ...
                 4.6321   44.6748    9.5109; ...
                 0.5227    4.6900   44.8061]);
end

if (matrixtype == 'rgb2xyz' | matrixtype == 'xyz2rgb')
  if (spacetype == 2)
    result = [16.9898   23.6831   15.0614; ...
               9.6167   45.8480    7.5094; ...
               0.9067    8.0767   78.2157];
  end
  if (spacetype == 10)
    result = [17.4665   27.7468   16.5398; ...
              10.0969   48.1835   11.6466; ...
               0.9293    7.3710   85.5683];
  end
  if (matrixtype == 'xyz2rgb')
    result = inv(result);
  end
end

