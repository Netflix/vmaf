% This is a simplified version of the S-CIELAB implementation by Xuemei
% Zhang, available at: http://white.stanford.edu/~brian/scielab/.
%
% Interface:
%           result = scielab_simple(sampPerDeg, image)
%
% Parameters:
%           result          Filtered image in XYZ coordinates
%           sampPerDeg      Samples per degree of the visual field       
%           image           Input image in XYZ coordinates
function result = scielab_simple(sampPerDeg, image)

%%%%%%%%%%%%%%%%%%%%%%%% XYZ -> Poirson & Wandell %%%%%%%%%%%%%%%%%%%%%%%%%

opp = changeColorSpace(image, cmatrix('xyz2opp'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Prepare filters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[k1, k2, k3] = separableFilters(sampPerDeg, 3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%  Spatial Filtering %%%%%%%%%%%%%%%%%%%%%%%%%%%%

p1 = separableConv(opp(:,:,1), k1, abs(k1));
p2 = separableConv(opp(:,:,2), k2, abs(k2));
p3 = separableConv(opp(:,:,3), k3, abs(k3));

%%%%%%%%%%%%%%%%%%%%%%%% Poirson & Wandell -> XYZ %%%%%%%%%%%%%%%%%%%%%%%%%

opp = cat(3, p1, p2, p3);
xyz = changeColorSpace(opp, cmatrix('opp2xyz'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Return Result %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

result = xyz;